"""
BlueCast Serve: One-command API deployment for trained pipelines.

Optional module -- install dependencies with:
    pip install bluecast[serve]

Usage::

    from bluecast.serve import serve, export_api

    # Quick local server
    serve(automl, port=8080)

    # Export standalone deployment
    export_api(automl, output_dir="./deployment")
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def serve(
    pipeline: Any,
    host: str = "0.0.0.0",
    port: int = 8080,
    log_level: str = "info",
) -> None:
    """Start a FastAPI server to serve predictions from a trained pipeline.

    :param pipeline: A trained BlueCast pipeline (any variant).
    :param host: Host address to bind to.
    :param port: Port to listen on.
    :param log_level: Uvicorn log level.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required to run the server. "
            "Install with: pip install 'bluecast[serve]'"
        )

    from bluecast.serve.app import create_app

    app = create_app(pipeline)

    print(f"Starting BlueCast Model API on http://{host}:{port}")
    print(f"Swagger docs: http://{host}:{port}/docs")
    print(f"Health check: http://{host}:{port}/health")

    uvicorn.run(app, host=host, port=port, log_level=log_level)


def export_api(
    pipeline: Any,
    output_dir: str,
    include_docker: bool = True,
) -> str:
    """Export a trained pipeline as a standalone FastAPI deployment.

    Creates a directory with ``app.py``, ``model.pkl``, ``requirements.txt``,
    an optional ``Dockerfile``, and ``README.md``. The generated app is
    self-contained and does not depend on ``bluecast.serve`` at runtime.

    :param pipeline: A trained BlueCast pipeline (any variant).
    :param output_dir: Path to the output directory.
    :param include_docker: Whether to generate a Dockerfile.
    :returns: Absolute path to the output directory.
    """
    from bluecast.serve.exporter import export_api as _export

    return _export(pipeline, output_dir, include_docker=include_docker)
