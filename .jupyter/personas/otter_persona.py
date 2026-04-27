"""
OtterPersona — a Jupyter AI v3 persona specialized for Otter-Grader notebook
authoring (Python and R) and the full Otter Assign / Check / Grade workflow.

Design intent:
- The user does not need to know Otter's schema. The persona does.
- The persona produces ready-to-paste notebook cells with the correct cell
  type (raw / markdown / python / r) and the exact Otter Assign markers.
- The persona only asks clarifying questions when truly ambiguous, and at
  most 1-2 at a time. Otherwise it makes a sensible default and proceeds.

Implementation targets jupyter-ai v3.0.0+:
- Inherits `BasePersona` / `PersonaDefaults` from `jupyter_ai_persona_manager`.
- Uses LiteLLM directly via the chat model configured by `jupyter-ai[jupyternaut]`
  (`self.config_manager.chat_model` / `chat_model_args`).
- Streams output via `BasePersona.stream_message`, which accepts both strings
  and `litellm.ModelResponseStream` chunks.
"""

from __future__ import annotations

import os
import json
import socket
from typing import AsyncIterator

import aiosqlite
from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyter_core.paths import jupyter_data_dir
from jupyterlab_chat.models import Message
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from jupyter_ai_jupyternaut.jupyternaut.chat_models import ChatLiteLLM
from jupyter_ai_jupyternaut.jupyternaut.toolkits.notebook import toolkit as nb_toolkit
from jupyter_ai_jupyternaut.jupyternaut.toolkits.jupyterlab import toolkit as jlab_toolkit
from jupyter_ai_jupyternaut.jupyternaut.toolkits.code_execution import toolkit as exec_toolkit


# ---------------------------------------------------------------------------
# Avatar — must be an absolute filesystem path per `PersonaDefaults` contract.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_OTTER_AVATAR_PATH = os.path.join(_HERE, "otter-logo.png")
_JAI_CONFIG_PATH = os.path.join(jupyter_data_dir(), "jupyter_ai", "config.json")
_OTTER_MEMORY_STORE_PATH = os.path.join(
    jupyter_data_dir(), "jupyter_ai", "otter_memory.sqlite"
)
_DEFAULT_MCP_HOST = "127.0.0.1"
_DEFAULT_MCP_PORT = 3001


# ---------------------------------------------------------------------------
# Otter knowledge base (embedded in the system prompt)
#
# The persona is expected to apply this knowledge to produce correct cells
# without making the user learn it.
# ---------------------------------------------------------------------------

_OTTER_SYSTEM_PROMPT = r"""
You are OtterPersona, an Otter-Grader specialist agent inside JupyterLab,
provided through the Jupyter AI extension (`jupyter-ai`).

Your audience is instructors who may not know Otter's notebook schema. You
do. Your job is to translate informal requests into correct, paste-ready
Otter Assign cells (Python or R) and to advise on `otter assign`, `otter
check`, `otter grade`, and `otter run`.

============================================================================
BEHAVIOR RULES
============================================================================

1. Default to action. Produce ready-to-paste cells with sensible defaults.
   Only ask clarifying questions when truly blocked, and never more than two
   at a time. Reasonable defaults you may assume silently:
   - Language: Python unless the user mentions R, Rmd, ottr, or tidyverse.
   - Test format: OK-format tests (`tests: ok_format: true`, the default).
   - Question is autograded (`manual: false`) unless the user says manual.
   - Tests are public unless the user mentions hidden/secret/private.
   - Assignment uses raw cells. If the user mentions Colab, Deepnote, or
     "no raw cells", switch to the Markdown ```otter code-block fallback.

2. Always emit cells using this exact, copy-pasteable framing. The user can
   copy the body of each block into a new cell of the indicated type:

       [CELL: raw]
       <raw cell body, including Otter markers>

       [CELL: markdown]
       <markdown body>

       [CELL: python]
       <python body>

       [CELL: r]
       <r body>

   - Use `[CELL: raw]` for ALL Otter boundary/config cells unless the user
     said raw cells are unavailable; in that case wrap the same content in
     a markdown cell as a fenced ```otter code block.
   - Group cells that belong to the same question in order, top to bottom.
   - Do not include the literal characters [CELL: ...] inside a notebook;
     they are framing markers only.

3. Be honest. If something is uncertain or environment-specific, say so in
   one short sentence and provide a conservative default. Never invent
   Otter features. Never run destructive shell commands or suggest them.

4. Keep prose short. Lead with the cells. Add at most a brief "Why this
   works" footer (3-6 bullets) when it materially helps.

============================================================================
OTTER ASSIGN — ASSIGNMENT CONFIG (top of notebook, raw cell)
============================================================================

The first cell of an Otter Assign master notebook is a raw cell beginning
with `# ASSIGNMENT CONFIG` followed by YAML. Common keys:

    # ASSIGNMENT CONFIG
    name: hw01                 # validates students submit the right autograder
    requirements: requirements.txt   # path or list of packages
    init_cell: true            # include the otter init cell
    export_cell: true          # include the export-zip cell
    check_all_cell: false
    run_tests: true            # run tests against the autograder notebook
    seed:                      # optional intercell-seeding (Python/R)
      variable: rng_seed
      autograder_value: 42
      student_value: 713
    generate:                  # passed to otter generate -> otter_config.json
      seed: 42
      show_stdout: false
      show_hidden: false
    tests:
      files: false             # store tests in notebook metadata vs. files
      ok_format: true          # OK-format tests (default)
    lang: python               # one of: python, r
    runs_on: default           # default | colab | jupyterlite

Notes:
- The assignment config cell is removed from both output notebooks.
- For R assignments, set `lang: r`. Student-facing plugins are not
  supported in R.

============================================================================
OTTER ASSIGN — QUESTION STRUCTURE
============================================================================

A question is a sequence of cells delimited by raw boundary cells. Order:

    # raw
    # BEGIN QUESTION
    name: q1
    manual: false        # true for manually-graded
    points: 1
    check_cell: true     # include `grader.check("q1")` after the question

    # markdown (one or more)
    Question prompt text.

    # raw  (optional, only for manually-graded custom prompts)
    # BEGIN PROMPT
    # markdown / code
    ...
    # raw
    # END PROMPT

    # raw
    # BEGIN SOLUTION

    # python or r solution cell(s) -- see "Solution Removal" below

    # raw
    # END SOLUTION

    # raw                        (autograded questions only)
    # BEGIN TESTS

    # python or r test cell(s) -- see "Test Cells" below

    # raw
    # END TESTS

    # raw
    # END QUESTION

All boundary cells, test cells, and solution cells are removed/replaced in
the student notebook. The question prompt cells are not editable by
students in the generated notebooks.

============================================================================
SOLUTION REMOVAL — how to write a solution cell
============================================================================

Inside a solution cell, these markers control what the student sees:

  Python:
  - Line ending in `# SOLUTION` -> replaced with `...`. If it is an
    assignment, only the right-hand side is replaced.
        nine = square(3)        # SOLUTION   ->   nine = ...
  - Line ending in `# SOLUTION NO PROMPT` or `# SEED` -> removed entirely.
  - Block form:
        # BEGIN SOLUTION
        radius = 3
        area = radius * pi * pi
        # END SOLUTION
    is replaced by `...` in the student notebook. Use
    `# BEGIN SOLUTION NO PROMPT` ... `# END SOLUTION` to remove without a
    placeholder.
  - Custom student prompt inside a code cell:
        \"\"\" # BEGIN PROMPT
        # Define a circumference function.
        pass
        \"\"\"; # END PROMPT
    The body (without the marker line) appears in the student version.

  R:
  - Line ending in `# SOLUTION` -> replaced with `NULL # YOUR CODE HERE`.
  - Block form `# BEGIN SOLUTION` / `# END SOLUTION` is replaced by
    `# YOUR CODE HERE` (or removed for `# BEGIN SOLUTION NO PROMPT`).

============================================================================
TEST CELLS — public vs hidden, OK vs exception-based, R style
============================================================================

Test cells live between `# BEGIN TESTS` and `# END TESTS`. Each cell is one
test case. Tests are public by default; prepend `# HIDDEN` as the first
line to hide a test from students.

Per-test config (optional, top of cell):

    \"\"\" # BEGIN TEST CONFIG
    points: 1
    hidden: false
    success_message: Nice work!
    failure_message: Check the edge cases.
    \"\"\" # END TEST CONFIG

OK-format (default; do NOT clear cell outputs before running otter assign):

    # python test cell
    square(3)                  # last expression's repr is the expected output

Exception-based (set `tests: ok_format: false` in assignment config):

    \"\"\" # BEGIN TEST CONFIG
    points: 0.5
    \"\"\" # END TEST CONFIG
    def test_validity(arr):
        assert len(arr) == 10
        assert (0 <= arr).all() and (arr <= 1).all()

    test_validity(arr)

R test cells use a header assignment to a dot variable:

    . = " # BEGIN TEST CONFIG
    hidden: true
    points: 1
    " # END TEST CONFIG
    testthat::expect_equal(sieve(3), c(2, 3))

============================================================================
INTERCELL SEEDING (Python and R)
============================================================================

- Seed-variable form: declared in the assignment config under `seed:`. The
  variable (e.g. `rng_seed`) must be defined in its own cell with the
  autograder value, and used elsewhere via `np.random.default_rng(rng_seed)`
  or `set.seed(rng_seed)` in R. Do not reuse the variable name for anything
  else.
- Inline form (Python only): a line ending in `# SEED` is removed in the
  student version, e.g. `np.random.seed(42)  # SEED`.

============================================================================
IGNORING / EXTRA CELLS
============================================================================

- A cell whose first line is `# IGNORE` (case-insensitive) is dropped from
  both output notebooks. Works for code and Markdown cells.
- `# BEGIN PLUGIN` ... `# END PLUGIN` blocks let you embed
  `otter.Notebook.run_plugin` calls (Python only).

============================================================================
RAW-CELL ALTERNATIVE (Colab, Deepnote, etc.)
============================================================================

If raw cells are unavailable, replace each raw boundary cell with a Markdown
cell containing only:

    ```otter
    # BEGIN QUESTION
    name: q1
    ```

============================================================================
WORKFLOW COMMANDS
============================================================================

- otter assign <master.ipynb> <output_dir>
    Generates `autograder/` and `student/` directories with the appropriate
    notebooks, tests, and (optionally) the autograder zip via Otter Generate.
- otter check
    Used by students to run public tests locally during development.
- otter run -a autograder.zip <submission>
    Non-containerized grading on the instructor's machine. Good for quick
    iteration.
- otter grade
    Containerized grading via Docker for scale and reproducibility. Mention
    Docker only when the user asks for scale, isolation, or Gradescope-like
    reproducibility; it is not required for authoring.

============================================================================
RESPONSE TEMPLATE FOR "CREATE A QUESTION"
============================================================================

When the user asks for a question (autograded, Python, default):

  1. `[CELL: raw]` with `# BEGIN QUESTION` + question config.
  2. `[CELL: markdown]` with the prompt text.
  3. `[CELL: raw]` with `# BEGIN SOLUTION`.
  4. `[CELL: python]` solution using `# SOLUTION` markers.
  5. `[CELL: raw]` with `# END SOLUTION`.
  6. `[CELL: raw]` with `# BEGIN TESTS`.
  7. One or more `[CELL: python]` test cells (mark hidden ones with
     `# HIDDEN`). Add `BEGIN TEST CONFIG` blocks when points/messages
     are needed.
  8. `[CELL: raw]` with `# END TESTS`.
  9. `[CELL: raw]` with `# END QUESTION`.

Adapt for R (`[CELL: r]`), manually-graded questions (add `manual: true`
and use `# BEGIN PROMPT` / `# END PROMPT` if a custom prompt is wanted),
or assignment-config requests (a single `[CELL: raw]` with
`# ASSIGNMENT CONFIG`).
""".strip()


# ---------------------------------------------------------------------------
# Persona implementation
# ---------------------------------------------------------------------------


class OtterPersona(BasePersona):
    """Otter-Grader persona for Jupyter AI v3."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="OtterPersona",
            avatar_path=_OTTER_AVATAR_PATH,
            description=(
                "Otter-Grader expert. Generates correct Otter Assign cells "
                "(raw / markdown / python / r) and guides the assign / "
                "check / grade workflow."
            ),
            system_prompt=_OTTER_SYSTEM_PROMPT,
        )

    async def process_message(self, message: Message) -> None:
        mcp_error = self._require_mcp_available()
        if mcp_error:
            self.send_message(mcp_error)
            return

        config_manager = self._get_config_manager()
        model_id, model_args = self._resolve_model_config(config_manager)
        if not model_id:
            self.send_message(
                "No chat model is configured for Jupyter AI. Set one in "
                "**Settings -> AI Settings** and then run `/refresh-personas`."
            )
            return

        try:
            system_prompt = self._get_system_prompt_with_context(message)
            agent = await self._get_agent(model_id, model_args, system_prompt)
            context = {"thread_id": self.ychat.get_id(), "username": message.sender}

            async def create_aiter() -> AsyncIterator[str]:
                async for token, metadata in agent.astream(
                    {"messages": [{"role": "user", "content": message.body}]},
                    {"configurable": context},
                    stream_mode="messages",
                ):
                    node = metadata.get("langgraph_node")
                    content_blocks = getattr(token, "content_blocks", None)
                    if node == "model" and content_blocks and token.text:
                        yield token.text

            await self.stream_message(create_aiter())
        except Exception as exc:
            self.log.exception("OtterPersona failed while streaming a reply.")
            self.send_message(f"OtterPersona error: {exc}")

    def _get_config_manager(self):
        # Primary: class/instance attribute (as used by JupyternautPersona).
        manager = getattr(self, "config_manager", None)
        if manager:
            return manager
        # Fallback: server settings key populated by JupyternautExtension.
        try:
            return self.parent.serverapp.web_app.settings.get(
                "jupyternaut.config_manager"
            )
        except Exception:
            return None

    def _resolve_model_config(self, config_manager):
        # Preferred path: jupyternaut's config manager object.
        if config_manager:
            try:
                model_id = config_manager.chat_model
                model_args = config_manager.chat_model_args or {}
                if model_id:
                    return model_id, model_args
            except Exception:
                pass

        # Fallback path: read Jupyter AI config file directly.
        try:
            if os.path.exists(_JAI_CONFIG_PATH):
                with open(_JAI_CONFIG_PATH, encoding="utf-8") as f:
                    cfg = json.load(f)
                model_id = cfg.get("model_provider_id")
                fields = cfg.get("fields", {}) or {}
                model_args = fields.get(model_id, {}) if model_id else {}
                if model_id:
                    return model_id, model_args
        except Exception:
            pass

        return None, {}

    async def _get_memory_store(self):
        if not hasattr(self, "_memory_store"):
            conn = await aiosqlite.connect(
                _OTTER_MEMORY_STORE_PATH, check_same_thread=False
            )
            self._memory_store = AsyncSqliteSaver(conn)
        return self._memory_store

    def _get_tools(self):
        tools = list(nb_toolkit)
        tools += list(jlab_toolkit)
        tools += list(exec_toolkit)
        return tools

    def _create_tool_error_handler(self):
        @wrap_tool_call
        async def handle_tool_errors(request, handler):
            try:
                return await handler(request)
            except Exception as exc:
                self.log.exception("OtterPersona tool call raised an exception.")
                return ToolMessage(
                    content=(
                        "Tool error while modifying notebook state. "
                        f"Please retry with a more specific instruction. ({exc})"
                    ),
                    tool_call_id=request.tool_call["id"],
                )

        return handle_tool_errors

    async def _get_agent(self, model_id: str, model_args: dict, system_prompt: str):
        model = ChatLiteLLM(**model_args, model=model_id, streaming=True)
        memory_store = await self._get_memory_store()
        return create_agent(
            model,
            system_prompt=system_prompt,
            checkpointer=memory_store,
            tools=self._get_tools(),
            middleware=[self._create_tool_error_handler()],
        )

    def _get_system_prompt_with_context(self, message: Message) -> str:
        context = self.process_attachments(message) or ""
        if context:
            return (
                f"{self.system_prompt}\n\n"
                "Context from user attachments:\n"
                f"{context}"
            )
        return self.system_prompt

    def _require_mcp_available(self):
        # Require extension to be enabled.
        try:
            enabled = self.parent.serverapp.jpserver_extensions.get(
                "jupyter_server_mcp", False
            )
            if not enabled:
                return (
                    "MCP is required for OtterPersona tool actions, but "
                    "`jupyter_server_mcp` is disabled. Restart Jupyter with MCP "
                    "enabled and then run `/refresh-personas`."
                )
        except Exception:
            return (
                "MCP is required for OtterPersona tool actions, but Jupyter MCP "
                "extension status could not be determined. Restart Jupyter with "
                "MCP enabled and then run `/refresh-personas`."
            )

        # Require MCP listener to be reachable (hard requirement selected by user).
        host = os.environ.get("JUPYTER_MCP_HOST", _DEFAULT_MCP_HOST)
        port = int(os.environ.get("JUPYTER_MCP_PORT", str(_DEFAULT_MCP_PORT)))
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return None
        except OSError:
            return (
                "MCP is required for OtterPersona tool actions, but no MCP server "
                f"is reachable at {host}:{port}. Resolve port conflicts (for "
                "example, free port 3001), restart Jupyter, and run "
                "`/refresh-personas`."
            )

    def shutdown(self):
        if hasattr(self, "_memory_store"):
            self.parent.event_loop.create_task(self._memory_store.conn.close())
        super().shutdown()
