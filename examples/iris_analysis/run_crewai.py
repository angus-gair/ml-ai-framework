"""Iris Classification Example using CrewAI Workflow.

This example demonstrates:
- Loading the Iris dataset
- Complete data analysis pipeline
- Multi-agent workflow orchestration
- Model training and evaluation
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.datasets import load_iris
import pandas as pd
from src.workflows.crew_system import CrewAIWorkflow
from src.utils.logging import setup_logging
from structlog import get_logger

# Setup logging
setup_logging(log_level="INFO", log_to_file=True, log_dir=str(project_root / "logs"))
logger = get_logger(__name__)


def prepare_iris_dataset(output_path: str = "/tmp/iris_dataset.csv"):
    """
    Load and save the Iris dataset.

    The Iris dataset contains:
    - 150 samples
    - 4 features (sepal length, sepal width, petal length, petal width)
    - 3 classes (setosa, versicolor, virginica)
    """
    logger.info("loading_iris_dataset")

    # Load Iris dataset from sklearn
    iris = load_iris()

    # Create DataFrame with feature names
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Add target column with species names
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    # Save to CSV for workflow
    df.to_csv(output_path, index=False)

    logger.info(
        "iris_dataset_prepared",
        samples=len(df),
        features=len(df.columns) - 1,
        classes=df['species'].nunique(),
        path=output_path
    )

    return output_path, df


def main():
    """Execute the Iris classification workflow using CrewAI."""
    print("=" * 80)
    print("Iris Classification - CrewAI Multi-Agent Workflow")
    print("=" * 80)
    print()

    # Step 1: Prepare the dataset
    print("Step 1: Preparing Iris dataset...")
    dataset_path, df = prepare_iris_dataset()
    print(f"✓ Dataset saved to: {dataset_path}")
    print(f"  - Samples: {len(df)}")
    print(f"  - Features: {len(df.columns) - 1}")
    print(f"  - Target: species (3 classes)")
    print()

    # Step 2: Initialize CrewAI workflow
    print("Step 2: Initializing CrewAI workflow...")
    print("  Creating 5 specialized agents:")
    print("    1. Data Loading Specialist")
    print("    2. Data Cleaning Specialist")
    print("    3. EDA Expert")
    print("    4. Machine Learning Engineer")
    print("    5. Report Analyst")
    print()

    workflow = CrewAIWorkflow(
        model_name="gpt-4",
        temperature=0.7,
        max_iterations=10,
    )

    # Step 3: Execute the workflow
    print("Step 3: Executing multi-agent workflow...")
    print("  This will run 5 sequential tasks:")
    print("    → Load dataset and extract metadata")
    print("    → Clean data and handle missing values")
    print("    → Perform exploratory data analysis")
    print("    → Train Random Forest classifier")
    print("    → Generate comprehensive report")
    print()
    print("Starting execution (this may take a few minutes)...")
    print("-" * 80)

    result = workflow.run(
        data_path=dataset_path,
        target_column="species",
    )

    # Step 4: Display results
    print()
    print("=" * 80)
    print("Workflow Execution Results")
    print("=" * 80)
    print()

    if result["status"] == "success":
        print("✓ Status: SUCCESS")
        print()
        print(f"Execution Time: {result['execution_time_seconds']:.2f} seconds")
        print(f"Agents Used: {result['agents_used']}")
        print(f"Tasks Completed: {result['tasks_completed']}")
        print(f"Timestamp: {result['timestamp']}")
        print()

        print("Final Report:")
        print("-" * 80)
        print(result['result'])
        print()

        # Save results
        output_file = project_root / "examples" / "iris_analysis" / "crewai_results.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("Iris Classification - CrewAI Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Execution Time: {result['execution_time_seconds']} seconds\n")
            f.write(f"Agents: {result['agents_used']}\n")
            f.write(f"Tasks: {result['tasks_completed']}\n")
            f.write(f"Timestamp: {result['timestamp']}\n\n")
            f.write("Final Report:\n")
            f.write("-" * 80 + "\n")
            f.write(str(result['result']))

        print(f"✓ Results saved to: {output_file}")

    else:
        print("✗ Status: FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Execution Time: {result['execution_time_seconds']:.2f} seconds")

    print()
    print("=" * 80)

    logger.info(
        "iris_crewai_workflow_completed",
        status=result["status"],
        duration=result["execution_time_seconds"]
    )


if __name__ == "__main__":
    main()
