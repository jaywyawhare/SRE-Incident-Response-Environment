from __future__ import annotations

import os

from api.server import app

__all__ = ["app", "main"]


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("SRE_PORT", "7860")))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
