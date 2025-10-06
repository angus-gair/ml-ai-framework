"""LangGraph workflow example."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflows.langgraph_system import LangGraphWorkflow
from src.utils.logging import setup_logging

# Setup logging
setup_logging(log_level="INFO", json_logs=False)


def main():
    """Run LangGraph workflow example."""
    # Initialize workflow
    workflow = LangGraphWorkflow(
        model_name="gpt-4",
        temperature=0.7,
    )

    # Example dataset path (replace with your actual dataset)
    data_path = "data/sample_dataset.csv"
    target_column = "target"

    print(f"Starting LangGraph ML workflow for: {data_path}")
    print(f"Target column: {target_column}")
    print("-" * 60)

    # Execute workflow
    result = workflow.run(
        data_path=data_path,
        target_column=target_column,
    )

    # Print results
    print("\n" + "=" * 60)
    print("WORKFLOW RESULTS")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Execution Time: {result['execution_time_seconds']:.2f} seconds")

    if 'report' in result:
        report = result['report']
        print("\nDATASET METADATA:")
        print(f"  Shape: {report['dataset_metadata']['shape']}")
        print(f"  Memory: {report['dataset_metadata']['memory_mb']:.2f} MB")

        print("\nCLEANING SUMMARY:")
        print(f"  Rows Removed: {report['cleaning_summary']['rows_removed']}")
        print(f"  Operations: {report['cleaning_summary']['operations_count']}")

        print("\nEDA SUMMARY:")
        print(f"  Features Analyzed: {report['eda_summary']['features_analyzed']}")
        print(f"  Outliers: {report['eda_summary']['outliers_detected']}")

        print("\nMODEL PERFORMANCE:")
        print(f"  Type: {report['model_performance']['model_type']}")
        print(f"  Algorithm: {report['model_performance']['algorithm']}")
        print(f"  Validation Score: {report['model_performance']['validation_score']:.4f}")

    if result['status'] == 'success':
        print("\n✅ Workflow completed successfully!")
    else:
        print(f"\n❌ Workflow failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
