"""Export a trained BlueCast pipeline as a standalone deployment directory."""

import json
import logging
import os
from typing import Any

from bluecast.general_utils.general_utils import save_to_production
from bluecast.serve.schemas import _extract_column_info, _get_class_problem

logger = logging.getLogger(__name__)


def export_api(
    pipeline: Any,
    output_dir: str,
    include_docker: bool = True,
) -> str:
    """Export a trained pipeline as a self-contained FastAPI deployment.

    Creates a directory with a standalone ``app.py``, serialized model,
    ``requirements.txt``, optional ``Dockerfile``, and a ``README.md``.
    The generated app does NOT depend on ``bluecast.serve`` at runtime —
    only on ``bluecast`` core, ``fastapi``, and ``uvicorn``.

    :param pipeline: A trained BlueCast pipeline (any variant).
    :param output_dir: Path to the output directory (created if needed).
    :param include_docker: Whether to generate a Dockerfile.
    :returns: The absolute path to the output directory.
    """
    try:
        from jinja2 import Environment, PackageLoader
    except ImportError:
        raise ImportError(
            "jinja2 is required for export_api. "
            "It should already be installed as a BlueCast dependency."
        )

    os.makedirs(output_dir, exist_ok=True)
    abs_output = os.path.abspath(output_dir)

    columns = _extract_column_info(pipeline)
    class_problem = _get_class_problem(pipeline)

    # Serialize the pipeline (save_to_production appends file_type to path)
    model_base = os.path.join(abs_output, "model")
    save_to_production(pipeline, model_base, file_type=".pkl")
    logger.info(f"Model saved to {model_base}.pkl")

    env = Environment(
        loader=PackageLoader("bluecast.serve", "templates"),
        keep_trailing_newline=True,
    )

    # Render app.py
    app_template = env.get_template("app_standalone.py.j2")
    columns_json = json.dumps(columns, indent=4)
    app_code = app_template.render(
        columns=columns,
        columns_json=columns_json,
        class_problem=class_problem,
    )
    _write(abs_output, "app.py", app_code)

    # Render requirements.txt
    requirements = "bluecast>=3.0.0\nfastapi>=0.100.0\nuvicorn>=0.20.0\ndill>=0.3.3\n"
    _write(abs_output, "requirements.txt", requirements)

    # Render Dockerfile
    if include_docker:
        dockerfile_template = env.get_template("Dockerfile.j2")
        _write(abs_output, "Dockerfile", dockerfile_template.render())

    # Render README
    example_payload = json.dumps(
        {
            col["name"]: 0.0 if col["python_type"] == "float" else "example"
            for col in columns[:5]
        },
        indent=2,
    )
    readme_template = env.get_template("README.md.j2")
    readme = readme_template.render(
        class_problem=class_problem,
        example_payload=example_payload,
    )
    _write(abs_output, "README.md", readme)

    logger.info(f"Deployment exported to {abs_output}")
    return abs_output


def _write(directory: str, filename: str, content: str) -> None:
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        f.write(content)
    logger.info(f"  Written: {path}")
