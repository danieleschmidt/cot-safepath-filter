"""
Server entry point for CoT SafePath Filter API.
"""

import uvicorn
from .api.app import app


def main() -> None:
    """Main server entry point."""
    uvicorn.run(
        "cot_safepath.api.app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()