"""
Model Serving: Deploy BlueCast Pipelines as REST APIs
======================================================

BlueCast can deploy trained pipelines as production REST APIs with
a single command. This is an optional module:

    pip install bluecast[serve]

This example demonstrates:
1. Training a model
2. Exporting a standalone deployment (app.py, Dockerfile, requirements.txt)
3. Starting a local API server

Endpoints:
    GET  /health         → health check
    GET  /schema         → input column schema
    GET  /metrics        → training metrics
    POST /predict        → single prediction
    POST /predict/batch  → batch predictions
    GET  /docs           → Swagger documentation
"""

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig

fast_config = TrainingConfig(
    hyperparameter_tuning_rounds=10,
    hyperparameter_tuning_max_runtime_secs=30,
    enable_feature_selection=False,
    calculate_shap_values=False,
    plot_hyperparameter_tuning_overview=False,
    hypertuning_cv_folds=2,
    train_size=0.8,
)


def make_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n, n_features=8, n_informative=6, random_state=seed
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(8)])
    df["category"] = rng.choice(["A", "B", "C"], size=n)
    df["target"] = y
    return df


# --- Train a model ---
print("=" * 60)
print("1. TRAINING A MODEL")
print("=" * 60)

df = make_data()
df_train, df_eval = train_test_split(df, test_size=0.2, random_state=42)
y_eval = df_eval.pop("target")

automl = BlueCast(class_problem="binary", conf_training=fast_config)
metrics = automl.fit_eval(df_train, df_eval, y_eval, target_col="target")
print(f"ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}")


# --- Export deployment ---
print("\n" + "=" * 60)
print("2. EXPORTING DEPLOYMENT")
print("=" * 60)

try:
    from bluecast.serve import export_api  # noqa: E402

    with tempfile.TemporaryDirectory() as tmpdir:
        output = export_api(automl, output_dir=tmpdir)
        print(f"\nDeployment exported to: {output}")
        print("Contents:")
        for f in sorted(os.listdir(output)):
            size = os.path.getsize(os.path.join(output, f))
            print(f"  {f:<25s} {size:>8,} bytes")

        # Show generated app.py header
        with open(os.path.join(output, "app.py")) as f:
            lines = f.readlines()
        print(f"\nGenerated app.py ({len(lines)} lines):")
        for line in lines[:15]:
            print(f"  {line}", end="")
        print("  ...")

        # Show Dockerfile
        with open(os.path.join(output, "Dockerfile")) as f:
            print(f"\nDockerfile:")
            print(f.read())

except ImportError:
    print("bluecast[serve] not installed. Showing usage instead:")
    print("""
    from bluecast.serve import serve, export_api

    # Export standalone deployment directory
    export_api(automl, output_dir="./deployment")

    # Or start a local server directly
    serve(automl, port=8080)
    # → http://localhost:8080/docs for Swagger UI
    """)


# --- Demonstrate schema detection ---
print("=" * 60)
print("3. SCHEMA AUTO-DETECTION")
print("=" * 60)

from bluecast.serve.schemas import (  # noqa: E402
    build_schema_response,
)

schema = build_schema_response(automl)
print(f"Problem type: {schema['class_problem']}")
print(f"Columns detected: {schema['n_columns']}")
for col in schema["columns"][:5]:
    print(f"  {col['name']:<20s} type={col['type']}")
if len(schema["columns"]) > 5:
    print(f"  ... and {len(schema['columns']) - 5} more")


# --- Local server (commented out - uncomment to run) ---
print("\n" + "=" * 60)
print("4. LOCAL SERVER (uncomment to run)")
print("=" * 60)
print("""
To start the server, uncomment these lines:

    from bluecast.serve import serve
    serve(automl, port=8080)

Then visit:
    http://localhost:8080/docs    → Swagger UI
    http://localhost:8080/health  → Health check
    http://localhost:8080/schema  → Input schema

Example curl:
    curl -X POST http://localhost:8080/predict \\
      -H "Content-Type: application/json" \\
      -d '{"feat_0": 0.5, "feat_1": -0.3, "category": "A"}'
""")

print("Serving example completed!")
