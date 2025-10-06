"""Iris Classification Example using LangGraph Workflow.

This example demonstrates:
- State-based workflow execution
- Graph-based agent coordination
- Sequential task processing with state management
- Iris dataset classification
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.datasets import load_iris
import pandas as pd
from src.workflows.langgraph_system import LangGraphWorkflow
from src.utils.logging import setup_logging
from structlog import get_logger
import json

# Setup logging
setup_logging(log_level="INFO", log_to_file=True, log_dir=str(project_root / "logs"))
logger = get_logger(__name__)


def prepare_iris_dataset(output_path: str = "/tmp/iris_dataset_langgraph.csv"):
    """
    Load and save the Iris dataset.

    The Iris dataset is a classic ML dataset with:
    - 150 samples
    - 4 numerical features
    - 3 balanced classes
    - No missing values
    """
    logger.info("loading_iris_dataset_for_langgraph")

    # Load Iris dataset
    iris = load_iris()

    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    # Save to CSV
    df.to_csv(output_path, index=False)

    logger.info(
        "iris_dataset_prepared_langgraph",
        path=output_path,
        shape=df.shape
    )

    return output_path, df


def main():
    """Execute the Iris classification workflow using LangGraph."""
    print("=" * 80)
    print("Iris Classification - LangGraph State-Based Workflow")
    print("=" * 80)
    print()

    # Step 1: Prepare dataset
    print("Step 1: Preparing Iris dataset...")
    dataset_path, df = prepare_iris_dataset()
    print(f"✓ Dataset ready: {dataset_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {', '.join(df.columns[:-1])}")
    print(f"  Target: species")
    print()

    # Step 2: Initialize LangGraph workflow
    print("Step 2: Building LangGraph workflow...")
    print("  Graph Structure:")
    print("    START")
    print("      ↓")
    print("    load_data")
    print("      ↓")
    print("    clean_data")
    print("      ↓")
    print("    perform_eda")
    print("      ↓")
    print("    train_model")
    print("      ↓")
    print("    generate_report")
    print("      ↓")
    print("    END")
    print()

    workflow = LangGraphWorkflow(
        model_name="gpt-4",
        temperature=0.7,
    )

    # Step 3: Execute workflow
    print("Step 3: Executing state-based workflow...")
    print("  Each node will update the shared state:")
    print("    - Dataset and metadata")
    print("    - Cleaned data and cleaning report")
    print("    - EDA findings")
    print("    - Trained model and performance metrics")
    print("    - Final analysis report")
    print()
    print("Starting execution...")
    print("-" * 80)

    result = workflow.run(
        data_path=dataset_path,
        target_column="species",
    )

    # Step 4: Display results
    print()
    print("=" * 80)
    print("LangGraph Workflow Results")
    print("=" * 80)
    print()

    if result["status"] in ["success", "completed_with_errors"]:
        print(f"✓ Status: {result['status'].upper()}")
        print()
        print(f"Execution Time: {result['execution_time_seconds']:.2f} seconds")
        print(f"Timestamp: {result['timestamp']}")
        print()

        # Display detailed report
        report = result.get("report", {})

        if "dataset_metadata" in report:
            print("Dataset Metadata:")
            print(f"  - Name: {report['dataset_metadata']['name']}")
            print(f"  - Shape: {report['dataset_metadata']['shape']}")
            print(f"  - Memory: {report['dataset_metadata']['memory_mb']} MB")
            print()

        if "cleaning_summary" in report:
            print("Cleaning Summary:")
            print(f"  - Rows removed: {report['cleaning_summary']['rows_removed']}")
            print(f"  - Operations: {report['cleaning_summary']['operations_count']}")
            print()

        if "eda_summary" in report:
            print("EDA Summary:")
            print(f"  - Features analyzed: {report['eda_summary']['features_analyzed']}")
            print(f"  - Outliers detected: {report['eda_summary']['outliers_detected']}")
            print(f"  - Correlations found: {report['eda_summary']['high_correlations']}")
            print()

        if "model_performance" in report:
            print("Model Performance:")
            print(f"  - Model Type: {report['model_performance']['model_type']}")
            print(f"  - Algorithm: {report['model_performance']['algorithm']}")
            print(f"  - Validation Score: {report['model_performance']['validation_score']:.4f}")
            print(f"  - Training Time: {report['model_performance']['training_time']:.2f}s")
            print()

        # Display execution messages
        if "messages" in result:
            print("Execution Messages:")
            for i, msg in enumerate(result["messages"], 1):
                print(f"  {i}. {msg}")
            print()

        # Save results
        output_file = project_root / "examples" / "iris_analysis" / "langgraph_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"✓ Results saved to: {output_file}")

    else:
        print("✗ Status: FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Execution Time: {result['execution_time_seconds']:.2f} seconds")

    print()
    print("=" * 80)

    logger.info(
        "iris_langgraph_workflow_completed",
        status=result["status"],
        duration=result["execution_time_seconds"]
    )


if __name__ == "__main__":
    main()
