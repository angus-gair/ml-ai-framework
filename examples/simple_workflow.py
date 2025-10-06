"""Simple workflow example using CrewAI."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflows.crew_system import CrewAIWorkflow
from src.utils.logging import setup_logging

# Setup logging
setup_logging(log_level="INFO", json_logs=False)


def main():
    """Run simple CrewAI workflow example."""
    # Initialize workflow
    workflow = CrewAIWorkflow(
        model_name="gpt-4",
        temperature=0.7,
    )

    # Example dataset path (replace with your actual dataset)
    data_path = "data/sample_dataset.csv"
    target_column = "target"

    print(f"Starting ML workflow for: {data_path}")
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
    print(f"Agents Used: {result['agents_used']}")
    print(f"Tasks Completed: {result['tasks_completed']}")

    if result['status'] == 'success':
        print("\n✅ Workflow completed successfully!")
    else:
        print(f"\n❌ Workflow failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
