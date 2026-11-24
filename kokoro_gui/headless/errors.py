"""The one error type the headless front ends raise and report.

The CLI turns it into an exit code and a stderr line (or a JSON object with
`--json`), the MCP server into an `isError` result. Both carry the same
`{error, message, valid}` shape from `to_json`.
"""
from __future__ import annotations

EXIT_OK = 0
EXIT_BAD_ARGUMENTS = 2
EXIT_ENGINE_NOT_INSTALLED = 3
EXIT_UNKNOWN_VOICE = 4
EXIT_MODEL_NOT_DOWNLOADED = 5
EXIT_PROJECT_LOCKED = 6
EXIT_GENERATION_FAILED = 7
EXIT_INTERRUPTED = 130

# error code -> CLI exit code. A code missing here exits with 2.
EXIT_CODES = {
    "bad_arguments": EXIT_BAD_ARGUMENTS,
    "out_not_allowed": EXIT_BAD_ARGUMENTS,
    "unsaved_session": EXIT_BAD_ARGUMENTS,
    "engine_not_installed": EXIT_ENGINE_NOT_INSTALLED,
    "unknown_voice": EXIT_UNKNOWN_VOICE,
    "model_not_downloaded": EXIT_MODEL_NOT_DOWNLOADED,
    "project_locked": EXIT_PROJECT_LOCKED,
    "generation_failed": EXIT_GENERATION_FAILED,
    "interrupted": EXIT_INTERRUPTED,
}


class HeadlessError(Exception):
    """A failure the caller can act on.

    `code` is a short stable string, `message` is for a person, `valid` lists
    the accepted values when the caller picked a wrong one (voices, engines).
    `exit_code` defaults from `EXIT_CODES`.
    """

    def __init__(self, code: str, message: str, valid=None, exit_code: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.valid = list(valid) if valid is not None else None
        self.exit_code = EXIT_CODES.get(code, EXIT_BAD_ARGUMENTS) if exit_code is None else exit_code

    def to_json(self) -> dict:
        return {"error": self.code, "message": self.message, "valid": self.valid}

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
