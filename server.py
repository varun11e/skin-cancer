"""Compatibility entry point.

The maintained Flask application lives in backend.app.
Run it with: python -m backend.app
"""

from backend.app import app


if __name__ == "__main__":
    from backend.config import Config
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)
