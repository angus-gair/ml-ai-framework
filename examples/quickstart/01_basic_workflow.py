"""Quick Start 01: Basic Workflow

This is the simplest possible ML workflow using the framework.
Learn the fundamentals in under 50 lines of code!

What you'll learn:
- Loading a dataset
- Running a complete ML workflow
- Interpreting results
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.datasets import load_iris
import pandas as pd
from src.workflows.langgraph_system import LangGraphWorkflow


def main():
    """Run the simplest possible ML workflow."""

    print("Quick Start 01: Basic Workflow")
    print("=" * 60)
    print()

    # Step 1: Prepare data
    print("Step 1: Loading Iris dataset...")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]

    # Save to file
    data_path = "/tmp/quickstart_iris.csv"
    df.to_csv(data_path, index=False)
    print(f"✓ Saved to: {data_path}")
    print(f"  Shape: {df.shape}")
    print()

    # Step 2: Create workflow
    print("Step 2: Creating LangGraph workflow...")
    workflow = LangGraphWorkflow(
        model_name="gpt-4",
        temperature=0.7,
    )
    print("✓ Workflow initialized")
    print()

    # Step 3: Run workflow
    print("Step 3: Running ML pipeline...")
    print("  (This will load → clean → analyze → train → report)")
    print()

    result = workflow.run(
        data_path=data_path,
        target_column="species",
    )

    # Step 4: View results
    print()
    print("Step 4: Results")
    print("=" * 60)

    if result["status"] == "success":
        report = result["report"]
        perf = report["model_performance"]

        print(f"✓ Status: {result['status']}")
        print(f"⏱  Time: {result['execution_time_seconds']:.2f}s")
        print()
        print(f"Model: {perf['algorithm']}")
        print(f"Accuracy: {perf['validation_score']:.2%}")
        print()
        print("That's it! You just ran a complete ML pipeline.")

    else:
        print(f"✗ Failed: {result.get('error')}")

    print()
    print("=" * 60)
    print()
    print("Next: Try 02_custom_tools.py to learn about custom tools")


if __name__ == "__main__":
    main()
