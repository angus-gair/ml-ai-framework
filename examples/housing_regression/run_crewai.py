"""California Housing Regression Example using CrewAI Workflow.

This example demonstrates:
- Regression task with real-world dataset
- Handling larger datasets (20k+ samples)
- Feature engineering insights
- Performance metrics for regression
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.datasets import fetch_california_housing
import pandas as pd
from src.workflows.crew_system import CrewAIWorkflow
from src.utils.logging import setup_logging
from structlog import get_logger

# Setup logging
setup_logging(log_level="INFO", log_to_file=True, log_dir=str(project_root / "logs"))
logger = get_logger(__name__)


def prepare_housing_dataset(output_path: str = "/tmp/california_housing.csv", sample_size: int = None):
    """
    Load and save the California Housing dataset.

    The California Housing dataset contains:
    - 20,640 samples from California districts (1990 census)
    - 8 features (median income, house age, rooms, bedrooms, population, etc.)
    - Target: median house value in $100,000s

    Args:
        output_path: Where to save the CSV file
        sample_size: If provided, randomly sample this many rows
    """
    logger.info("loading_california_housing_dataset")

    # Load California Housing dataset
    housing = fetch_california_housing()

    # Create DataFrame with feature names
    df = pd.DataFrame(housing.data, columns=housing.feature_names)

    # Add target column (median house value)
    df['MedHouseVal'] = housing.target

    # Optional: Sample for faster execution
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        logger.info("dataset_sampled", original_size=len(housing.data), sample_size=sample_size)

    # Save to CSV
    df.to_csv(output_path, index=False)

    logger.info(
        "housing_dataset_prepared",
        samples=len(df),
        features=len(df.columns) - 1,
        target_mean=df['MedHouseVal'].mean(),
        path=output_path
    )

    return output_path, df


def main():
    """Execute the housing regression workflow using CrewAI."""
    print("=" * 80)
    print("California Housing Regression - CrewAI Multi-Agent Workflow")
    print("=" * 80)
    print()

    # Step 1: Prepare the dataset
    print("Step 1: Preparing California Housing dataset...")
    print("  Note: Using full dataset (20,640 samples)")
    print("  For faster execution, you can modify sample_size parameter")
    print()

    dataset_path, df = prepare_housing_dataset()

    print(f"✓ Dataset saved to: {dataset_path}")
    print(f"  - Samples: {len(df)}")
    print(f"  - Features: {len(df.columns) - 1}")
    print(f"  - Target: MedHouseVal (median house value in $100k)")
    print()
    print("  Feature Overview:")
    for col in df.columns[:-1]:
        print(f"    • {col}: {df[col].min():.2f} - {df[col].max():.2f}")
    print()
    print(f"  Target Statistics:")
    print(f"    • Mean: ${df['MedHouseVal'].mean() * 100000:.0f}")
    print(f"    • Median: ${df['MedHouseVal'].median() * 100000:.0f}")
    print(f"    • Range: ${df['MedHouseVal'].min() * 100000:.0f} - ${df['MedHouseVal'].max() * 100000:.0f}")
    print()

    # Step 2: Initialize CrewAI workflow
    print("Step 2: Initializing CrewAI regression workflow...")
    print("  Creating specialized agents for regression task:")
    print("    1. Data Loading Specialist - Handles large datasets")
    print("    2. Data Cleaning Specialist - Outlier detection critical for housing")
    print("    3. EDA Expert - Feature correlations and distributions")
    print("    4. ML Engineer - Regression model training")
    print("    5. Report Analyst - Performance metrics (R², MSE, MAE)")
    print()

    workflow = CrewAIWorkflow(
        model_name="gpt-4",
        temperature=0.7,
        max_iterations=15,  # More iterations for larger dataset
    )

    # Step 3: Execute the workflow
    print("Step 3: Executing multi-agent regression workflow...")
    print("  Pipeline stages:")
    print("    → Load 20k+ samples with metadata extraction")
    print("    → Clean data (handle outliers, missing values)")
    print("    → EDA (correlations, distributions, feature importance)")
    print("    → Train Random Forest Regressor")
    print("    → Evaluate with R², RMSE, MAE metrics")
    print()
    print("Starting execution (this may take 2-5 minutes for full dataset)...")
    print("-" * 80)

    result = workflow.run(
        data_path=dataset_path,
        target_column="MedHouseVal",
    )

    # Step 4: Display results
    print()
    print("=" * 80)
    print("Regression Workflow Results")
    print("=" * 80)
    print()

    if result["status"] == "success":
        print("✓ Status: SUCCESS")
        print()
        print(f"⏱  Execution Time: {result['execution_time_seconds']:.2f} seconds")
        print(f"🤖 Agents Used: {result['agents_used']}")
        print(f"✅ Tasks Completed: {result['tasks_completed']}")
        print(f"📅 Timestamp: {result['timestamp']}")
        print()

        print("📊 Final Analysis Report:")
        print("-" * 80)
        print(result['result'])
        print()

        # Save results
        output_file = project_root / "examples" / "housing_regression" / "crewai_results.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("California Housing Regression - CrewAI Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Dataset: California Housing (1990 census)\n")
            f.write(f"Samples: {len(df)}\n")
            f.write(f"Features: {len(df.columns) - 1}\n")
            f.write(f"Target: Median House Value\n\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Execution Time: {result['execution_time_seconds']} seconds\n")
            f.write(f"Agents: {result['agents_used']}\n")
            f.write(f"Tasks: {result['tasks_completed']}\n")
            f.write(f"Timestamp: {result['timestamp']}\n\n")
            f.write("Performance Metrics Expected:\n")
            f.write("  - R² Score: 0.75-0.85 (on validation set)\n")
            f.write("  - RMSE: ~0.5-0.6 ($50k-60k error)\n")
            f.write("  - MAE: ~0.35-0.45 ($35k-45k error)\n\n")
            f.write("Final Report:\n")
            f.write("-" * 80 + "\n")
            f.write(str(result['result']))

        print(f"✓ Detailed results saved to: {output_file}")

    else:
        print("✗ Status: FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Execution Time: {result['execution_time_seconds']:.2f} seconds")

    print()
    print("=" * 80)
    print()
    print("💡 Key Insights for Housing Regression:")
    print("  1. MedInc (median income) is typically the strongest predictor")
    print("  2. Location features (Latitude, Longitude) show strong correlations")
    print("  3. HouseAge and AveRooms provide additional predictive power")
    print("  4. Outliers in house values can significantly impact RMSE")
    print()

    logger.info(
        "housing_crewai_workflow_completed",
        status=result["status"],
        duration=result["execution_time_seconds"]
    )


if __name__ == "__main__":
    main()
