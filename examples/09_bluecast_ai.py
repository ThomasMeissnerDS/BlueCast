"""
BlueCastAI: Multi-Agent LLM-Powered AutoML
============================================

BlueCastAI is an optional module that uses a multi-agent system to
analyze your data, engineer features, and build optimized BlueCast
pipelines -- all guided by natural language prompts.

Architecture:
    PlannerAgent       → interprets your prompt, creates execution plan
    DataAnalystAgent   → profiles data, checks correlations & leakage
    FeatureEngineerAgent → creates ratio, interaction, binned features
    PipelineBuilderAgent → configures and runs BlueCast pipelines
    EvaluatorAgent     → analyzes results, suggests improvements
    ResearcherAgent    → web search for techniques (optional)

Prerequisites:
    pip install bluecast[ai-gemini]     # Google Gemini
    pip install bluecast[ai-openai]     # OpenAI GPT-4
    pip install bluecast[ai-anthropic]  # Anthropic Claude
    pip install bluecast[ai]            # all three

Set your API key as an environment variable:
    export GEMINI_API_KEY="your-key"
    export OPENAI_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"

This example demonstrates:
1. Fast mode  — baseline model, no FE, 1 iteration (~2 min)
2. Balanced mode — targeted FE, stacking, 3 iterations (~10 min)
3. Precise mode — extensive FE, hill climbing, 5 iterations (~30 min)
4. Using context files for domain knowledge
5. Inspecting agent logs
6. Exporting reproducible pipeline code
"""

import os
import sys
import tempfile
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


# ---------------------------------------------------------------------------
# Create a realistic synthetic dataset
# ---------------------------------------------------------------------------
def make_credit_data(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        random_state=seed,
        flip_y=0.07,
    )
    df = pd.DataFrame(
        X,
        columns=[
            "income",
            "credit_score",
            "age",
            "debt_ratio",
            "num_accounts",
            "utilization",
            "payment_history",
            "inquiries",
            "credit_age",
            "balance",
        ],
    )
    df["income"] = (df["income"] * 15000 + 55000).clip(15000, 200000).round(0)
    df["credit_score"] = (df["credit_score"] * 50 + 700).clip(300, 850).round(0)
    df["age"] = (df["age"] * 10 + 42).clip(18, 75).round(0)
    df["employment"] = rng.choice(["employed", "self_employed", "retired"], size=n)
    df["region"] = rng.choice(["north", "south", "east", "west"], size=n)
    # Inject some nulls
    for col in ["income", "credit_score"]:
        df.loc[rng.random(n) < 0.03, col] = np.nan
    df["default"] = y
    return df


# ---------------------------------------------------------------------------
# Resolve API key and provider
# ---------------------------------------------------------------------------
API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY")
    or ""
)

PROVIDER: Literal["gemini", "openai", "anthropic"] = "gemini"
if os.environ.get("GEMINI_API_KEY"):
    PROVIDER = "gemini"
elif os.environ.get("OPENAI_API_KEY"):
    PROVIDER = "openai"
elif os.environ.get("ANTHROPIC_API_KEY"):
    PROVIDER = "anthropic"


# ---------------------------------------------------------------------------
# If no API key, show comprehensive usage examples and exit
# ---------------------------------------------------------------------------
if not API_KEY:
    df = make_credit_data(500)
    print("=" * 70)
    print(" BlueCastAI — Usage Examples")
    print(" (set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY to run live)")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│  EXAMPLE 1: Fast Mode — baseline model in ~2 minutes              │
└─────────────────────────────────────────────────────────────────────┘

    from bluecast.ai import BlueCastAI

    ai = BlueCastAI(api_key="your-key", provider="gemini")
    result = ai.run(
        df_train,
        target_col="default",
        prompt="Build a fast baseline model",
        mode="fast",           # skip FE, 1 iteration
    )
    result.show_report()
    y_probs, y_classes = result.predict(df_test)

┌─────────────────────────────────────────────────────────────────────┐
│  EXAMPLE 2: Balanced Mode — with feature engineering + stacking    │
└─────────────────────────────────────────────────────────────────────┘

    result = ai.run(
        df_train,
        target_col="default",
        prompt="Build a binary classifier with good feature engineering",
        mode="balanced",       # targeted FE, 3 iterations
    )

┌─────────────────────────────────────────────────────────────────────┐
│  EXAMPLE 3: Free Tier Rate Limit Handling                         │
└─────────────────────────────────────────────────────────────────────┘

    # LLM providers (Gemini, OpenAI, Anthropic) automatically use exponential
    # backoff to handle 429 Too Many Requests errors natively.
    # You can also introduce a sleep constraint between sequential LLM calls:
    ai = BlueCastAI(api_key="...", provider="gemini", llm_sleep_time=10)
    result = ai.run(
        df_train,
        target_col="default",
        prompt="Build a good model within rate limits",
    )

┌─────────────────────────────────────────────────────────────────────┐
│  EXAMPLE 4: Precise Mode — maximum performance                    │
└─────────────────────────────────────────────────────────────────────┘

    result = ai.run(
        df_train,
        target_col="default",
        prompt="Maximize ROC AUC using hill climbing ensemble, extensive "
               "feature engineering, and thorough hyperparameter tuning",
        mode="precise",        # full FE, 5+ iterations, hill climbing
        max_iterations=5,
    )

┌─────────────────────────────────────────────────────────────────────┐
│  EXAMPLE 5: With domain context files                             │
└─────────────────────────────────────────────────────────────────────┘

    result = ai.run(
        df_train,
        target_col="default",
        prompt="Build a credit risk model following regulatory guidelines",
        context_files=[
            "data_dictionary.txt",    # column descriptions
            "business_rules.md",      # domain constraints
        ],
    )

┌─────────────────────────────────────────────────────────────────────┐
│  EXAMPLE 6: Checkpointing — resume after crashes                  │
└─────────────────────────────────────────────────────────────────────┘

    # If a long run crashes, it resumes from the last completed step
    ai = BlueCastAI(api_key="...", checkpoint_dir="/tmp/my_run")
    result = ai.run(
        df_train,
        target_col="default",
        prompt="Build a precise model",
        mode="precise",
        max_iterations=5,
    )
    # If this crashes at step 4, re-run the same code —
    # steps 1-3 are skipped, execution resumes at step 4.

┌─────────────────────────────────────────────────────────────────────┐
│  EXAMPLE 7: Smart sampling for large datasets                     │
└─────────────────────────────────────────────────────────────────────┘

    # 1M-row dataset: agents analyze a 50k stratified sample,
    # but the final model trains on the full dataset.
    ai = BlueCastAI(api_key="...", provider="gemini")
    result = ai.run(
        df_1_million_rows,
        target_col="target",
        prompt="Build a good model",
    )

    # Custom configuration via AIConfig (tokens budget, UI callbacks, etc.):
    from bluecast.ai.config import AIConfig

    def my_callback(agent, message, event_type, metadata):
        print(f"UI update from {agent}: {message[:30]}")

    config = AIConfig(
        api_key="...",
        max_rows_for_agents=20_000,
        max_tokens_budget=50_000,
        callbacks=[my_callback]
    )

┌─────────────────────────────────────────────────────────────────────┐
│  EXAMPLE 8: Inspecting results + reports                          │
└─────────────────────────────────────────────────────────────────────┘

    # LLM-written Markdown report (from the Reporter agent)
    result.show_report()
    result.save_report("report.md")

    # Export reproducible Python code
    result.save_code("my_pipeline.py")

    # Full structured agent log (every tool call, every response)
    result.save_log("agent_log.json")

    # Trained pipeline — ready for production
    predictions = result.predict(df_test)

    # Detailed metrics
    print(result.metrics)

    # Raw Markdown report
    print(result.report_markdown)

    # Feature engineering code generated
    print(result.feature_engineering_code)

    # Structured log with timestamps, event types, metadata
    for entry in result.structured_log:
        print(f"[{entry.agent}] ({entry.event_type}) {entry.content[:80]}")

┌─────────────────────────────────────────────────────────────────────┐
│  EXAMPLE 9: Choosing your LLM provider                            │
└─────────────────────────────────────────────────────────────────────┘

    # Google Gemini (default model: gemini-2.5-flash)
    ai = BlueCastAI(api_key="...", provider="gemini")

    # Choose a specific Gemini model
    ai = BlueCastAI(api_key="...", provider="gemini", model="gemini-2.5-pro")
    ai = BlueCastAI(api_key="...", provider="gemini", model="gemini-2.0-flash-lite")

    # OpenAI (default model: gpt-4o)
    ai = BlueCastAI(api_key="...", provider="openai")
    ai = BlueCastAI(api_key="...", provider="openai", model="gpt-4o-mini")

    # Anthropic (default model: claude-sonnet-4-20250514)
    ai = BlueCastAI(api_key="...", provider="anthropic")
    ai = BlueCastAI(api_key="...", provider="anthropic", model="claude-opus-4-20250514")

    # The model= parameter accepts any model name your provider supports.
    # When omitted, defaults are:
    #   gemini    -> gemini-2.5-flash
    #   openai    -> gpt-4o
    #   anthropic -> claude-sonnet-4-20250514

    # Lower temperature for more deterministic results
    ai = BlueCastAI(api_key="...", provider="gemini", temperature=0.1)

┌─────────────────────────────────────────────────────────────────────┐
│  AGENT FLOW                                                       │
└─────────────────────────────────────────────────────────────────────┘

    User: "Build a precise binary classifier with hill climbing"
        │
        ▼
    Step 0: Smart Sampling (50k rows if dataset is large)
        │
        ▼
    Step 1: PlannerAgent → {problem: binary, ensemble: hill_climbing, FE: true}
        │                                                    [checkpoint]
        ▼
    Step 2: ResearcherAgent → web search (optional)
        │                                                    [checkpoint]
        ▼
    Step 3: DataAnalystAgent → profiles data, correlations, leakage
        │                                                    [checkpoint]
        ▼
    Step 4: FeatureEngineerAgent → ratios, interactions, bins
        │                                                    [checkpoint]
        ▼
    Step 5: Build → Evaluate → Improve Loop
        │   PipelineBuilder → builds & runs model
        │   Evaluator → analyzes, suggests changes
        │   (repeat up to max_iterations)                    [checkpoint]
        │
        ▼
    Step 6: ReporterAgent → writes polished Markdown report
        │                                                    [checkpoint]
        ▼
    BlueCastAIResult
    ├── .pipeline              → trained model
    ├── .predict(df)           → predictions
    ├── .save_code(path)       → reproducible .py
    ├── .save_report(path)     → Markdown report
    ├── .save_log(path)        → structured JSON log
    ├── .show_report()         → print report
    ├── .metrics               → evaluation scores
    ├── .report_markdown       → raw Markdown
    ├── .structured_log        → list of AgentLogEntry
    └── .agent_log             → flat string log
""")

    # Demonstrate that the module imports work even without provider SDKs
    try:
        from bluecast.ai.agents.reporter import ReporterAgent  # noqa: E402, F401
        from bluecast.ai.config import AIConfig  # noqa: E402, F401
        from bluecast.ai.context import AgentLogEntry, SharedContext  # noqa: E402, F401
        from bluecast.ai.result import BlueCastAIResult  # noqa: E402, F401
        from bluecast.ai.tools import (  # noqa: E402, F401
            TOOL_DEFINITIONS,
            tool_describe_data,
        )

        print("Module imports verified (no provider SDK needed for import).")
    except ImportError as e:
        print(f"Import note: {e}")

    ctx = SharedContext(
        df_train=df,
        target_col="default",
        user_prompt="test",
        original_shape=df.shape,
    )
    print("\nData summary preview (auto-generated for LLM):\n")
    print(ctx.get_data_summary()[:800])
    print("...")

    print(f"\nAvailable agent tools ({len(TOOL_DEFINITIONS)}):")
    for name, td in TOOL_DEFINITIONS.items():
        print(f"  {name}: {td.description[:70]}...")

    ctx.log(
        "Demo",
        "This is a structured log entry",
        event_type="info",
        metadata={"demo": True},
    )
    print("\nStructured log example:")
    print(f"  {ctx.structured_log[-1]}")

    print("\nAIConfig fields for extensive customization:")
    print("  max_rows_for_agents (default):    50,000")
    print("  max_columns_for_agents (default): 200")
    print("  checkpoint_dir (default):         None (disabled)")
    print("  max_tokens_budget (default):      None (no limit)")
    print("  callbacks (default):              [] (no external hooks)")

    print("\nSet an API key environment variable and re-run to see the live demo!")
    sys.exit(0)


# ===========================================================================
# LIVE DEMO (runs when API key is available)
# ===========================================================================

from bluecast.ai import BlueCastAI  # noqa: E402

df = make_credit_data()
print("=" * 70)
print(f" BlueCastAI Live Demo  (provider={PROVIDER})")
print("=" * 70)
print(f" Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f" Target:  'default' — {df['default'].value_counts().to_dict()}")
print()


# --- Run 1: Fast mode ---
print("-" * 70)
print(" Run 1: FAST MODE")
print("-" * 70)

ai = BlueCastAI(api_key=API_KEY, provider=PROVIDER)
result_fast = ai.run(
    df,
    target_col="default",
    prompt="Build a fast baseline binary classifier",
    mode="fast",
)

print(f"\nFast mode metrics: {result_fast.metrics}")
print(f"Agent log entries: {len(result_fast.agent_log)}")
print(f"Report generated: {'yes' if result_fast.report_markdown else 'no'}")


# --- Run 2: Balanced mode with checkpointing ---
print("\n" + "-" * 70)
print(" Run 2: BALANCED MODE (FE + stacking + checkpointing)")
print("-" * 70)

with tempfile.TemporaryDirectory() as checkpoint_dir:
    ai_ckpt = BlueCastAI(
        api_key=API_KEY,
        provider=PROVIDER,
        checkpoint_dir=checkpoint_dir,  # enables checkpoint save/resume
    )
    result_balanced = ai_ckpt.run(
        df,
        target_col="default",
        prompt="Build a good binary classifier with feature engineering and stacking",
        mode="balanced",
        max_iterations=2,
    )

print(f"\nBalanced mode metrics: {result_balanced.metrics}")

if result_balanced.feature_engineering_code:
    print("\nGenerated feature engineering code:")
    print(result_balanced.feature_engineering_code[:500])


# --- Inspect the result ---
print("\n" + "-" * 70)
print(" RESULT INSPECTION")
print("-" * 70)

# The Reporter agent's Markdown report
result_balanced.show_report()

# Export all artifacts
with tempfile.TemporaryDirectory() as tmpdir:
    # Pipeline code
    code_path = os.path.join(tmpdir, "generated_pipeline.py")
    result_balanced.save_code(code_path)
    with open(code_path) as f:
        code = f.read()
    print(f"\nGenerated pipeline code ({len(code)} chars):")
    print(code[:600])
    print("...")

    # Markdown report
    report_path = os.path.join(tmpdir, "report.md")
    result_balanced.save_report(report_path)
    print(f"\nReport saved: {report_path}")

    # Structured agent log
    log_path = os.path.join(tmpdir, "agent_log.json")
    result_balanced.save_log(log_path)
    print(f"Agent log saved: {log_path}")

# Predict
if result_balanced.pipeline is not None:
    preds = result_balanced.predict(df.drop("default", axis=1))
    if isinstance(preds, tuple):
        print(f"\nPredictions: {preds[0].shape[0]} samples")
    else:
        print(f"\nPredictions: {preds.shape[0]} samples")

# Structured log inspection
print(f"\nStructured log ({len(result_balanced.structured_log)} entries):")
for entry in result_balanced.structured_log[:10]:
    print(f"  [{entry.agent}] ({entry.event_type}) {entry.content[:80]}")
if len(result_balanced.structured_log) > 10:
    print(f"  ... and {len(result_balanced.structured_log) - 10} more")

print("\n" + "=" * 70)
print(" BlueCastAI demo complete!")
print("=" * 70)
