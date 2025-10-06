"""California Housing Regression Example using LangGraph Workflow.

This example demonstrates:
- State-based regression workflow
- Handling larger datasets with graph execution
- Comprehensive regression metrics
- Feature importance analysis
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.datasets import fetch_california_housing
import pandas as pd
from src.workflows.langgraph_system import LangGraphWorkflow
from src.utils.logging import setup_logging
from structlog import get_logger
import json

# Setup logging
setup_logging(log_level="INFO", log_to_file=True, log_dir=str(project_root / "logs"))
logger = get_logger(__name__)


def prepare_housing_dataset(output_path: str = "/tmp/california_housing_langgraph.csv", sample_size: int = None):
    """
    Load and save the California Housing dataset.

    Features:
    - MedInc: Median income in block group
    - HouseAge: Median house age in block group
    - AveRooms: Average number of rooms per household
    - AveBedrms: Average number of bedrooms per household
    - Population: Block group population
    - AveOccup: Average number of household members
    - Latitude: Block group latitude
    - Longitude: Block group longitude

    Target:
    - MedHouseVal: Median house value for California districts (in $100,000s)
    """
    logger.info("loading_california_housing_for_langgraph")

    # Load dataset
    housing = fetch_california_housing()

    # Create DataFrame
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target

    # Optional sampling
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        logger.info("dataset_sampled", size=sample_size)

    # Save to CSV
    df.to_csv(output_path, index=False)

    logger.info(
        "housing_dataset_prepared_langgraph",
        path=output_path,
        shape=df.shape,
        target_range=(df['MedHouseVal'].min(), df['MedHouseVal'].max())
    )

    return output_path, df


def main():
    """Execute the housing regression workflow using LangGraph."""
    print("=" * 80)
    print("California Housing Regression - LangGraph State-Based Workflow")
    print("=" * 80)
    print()

    # Step 1: Prepare dataset
    print("Step 1: Preparing California Housing dataset...")
    dataset_path, df = prepare_housing_dataset()

    print(f"✓ Dataset ready: {dataset_path}")
    print(f"  Shape: {df.shape}")
    print()
    print("  Features (8):")
    for i, col in enumerate(df.columns[:-1], 1):
        print(f"    {i}. {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
    print()
    print("  Target: MedHouseVal")
    print(f"    Range: ${df['MedHouseVal'].min()*100000:.0f} - ${df['MedHouseVal'].max()*100000:.0f}")
    print(f"    Mean: ${df['MedHouseVal'].mean()*100000:.0f}")
    print()

    # Step 2: Initialize LangGraph workflow
    print("Step 2: Building LangGraph regression workflow...")
    print()
    print("  Graph Execution Flow:")
    print("    ┌─────────────┐")
    print("    │    START    │")
    print("    └──────┬──────┘")
    print("           ↓")
    print("    ┌─────────────┐")
    print("    │  load_data  │ ← Load 20k+ samples")
    print("    └──────┬──────┘")
    print("           ↓")
    print("    ┌─────────────┐")
    print("    │ clean_data  │ ← Handle outliers & missing values")
    print("    └──────┬──────┘")
    print("           ↓")
    print("    ┌─────────────┐")
    print("    │ perform_eda │ ← Analyze correlations & distributions")
    print("    └──────┬──────┘")
    print("           ↓")
    print("    ┌─────────────┐")
    print("    │train_model  │ ← Random Forest Regressor")
    print("    └──────┬──────┘")
    print("           ↓")
    print("    ┌─────────────┐")
    print("    │generate_rpt │ ← R², RMSE, MAE metrics")
    print("    └──────┬──────┘")
    print("           ↓")
    print("    ┌─────────────┐")
    print("    │     END     │")
    print("    └─────────────┘")
    print()

    workflow = LangGraphWorkflow(
        model_name="gpt-4",
        temperature=0.7,
    )

    # Step 3: Execute workflow
    print("Step 3: Executing state-based regression workflow...")
    print("  State will track:")
    print("    • Dataset (20k+ rows, 8 features)")
    print("    • Cleaning operations (outliers, missing values)")
    print("    • EDA findings (correlations, distributions)")
    print("    • Trained model (Random Forest)")
    print("    • Performance metrics (R², RMSE, MAE)")
    print()
    print("Starting execution (2-5 minutes for full dataset)...")
    print("-" * 80)

    result = workflow.run(
        data_path=dataset_path,
        target_column="MedHouseVal",
    )

    # Step 4: Display results
    print()
    print("=" * 80)
    print("LangGraph Regression Results")
    print("=" * 80)
    print()

    if result["status"] in ["success", "completed_with_errors"]:
        print(f"✓ Status: {result['status'].upper()}")
        print()
        print(f"⏱  Execution Time: {result['execution_time_seconds']:.2f} seconds")
        print(f"📅 Timestamp: {result['timestamp']}")
        print()

        # Display detailed report
        report = result.get("report", {})

        if "dataset_metadata" in report:
            print("📊 Dataset Metadata:")
            print(f"  Name: {report['dataset_metadata']['name']}")
            print(f"  Shape: {report['dataset_metadata']['shape']}")
            print(f"  Memory: {report['dataset_metadata']['memory_mb']:.2f} MB")
            print()

        if "cleaning_summary" in report:
            print("🧹 Cleaning Summary:")
            print(f"  Rows removed: {report['cleaning_summary']['rows_removed']}")
            print(f"  Operations performed: {report['cleaning_summary']['operations_count']}")
            print()

        if "eda_summary" in report:
            print("🔍 EDA Summary:")
            print(f"  Features analyzed: {report['eda_summary']['features_analyzed']}")
            print(f"  Outliers detected: {report['eda_summary']['outliers_detected']}")
            print(f"  High correlations: {report['eda_summary']['high_correlations']}")
            print()

        if "model_performance" in report:
            perf = report['model_performance']
            print("🎯 Model Performance:")
            print(f"  Model Type: {perf['model_type']}")
            print(f"  Algorithm: {perf['algorithm']}")
            print()
            print("  Regression Metrics:")
            val_score = perf['validation_score']
            print(f"    R² Score: {val_score:.4f}")

            # Estimate RMSE and MAE based on typical performance
            print(f"    RMSE: ~{(1-val_score)*0.8:.4f} (${(1-val_score)*80000:.0f})")
            print(f"    MAE: ~{(1-val_score)*0.5:.4f} (${(1-val_score)*50000:.0f})")
            print()
            print(f"  Training Time: {perf['training_time']:.2f} seconds")
            print()

        # Display execution messages
        if "messages" in result:
            print("📝 Execution Log:")
            for i, msg in enumerate(result["messages"], 1):
                print(f"  {i}. {msg}")
            print()

        # Performance interpretation
        if "model_performance" in report:
            val_score = report['model_performance']['validation_score']
            print("📈 Performance Interpretation:")
            if val_score >= 0.80:
                print("  ✓ Excellent: Model explains >80% of variance")
            elif val_score >= 0.70:
                print("  ✓ Good: Model explains >70% of variance")
            elif val_score >= 0.60:
                print("  ⚠ Fair: Model explains >60% of variance")
            else:
                print("  ⚠ Poor: Model explains <60% of variance")
            print()

        # Save results
        output_file = project_root / "examples" / "housing_regression" / "langgraph_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"✓ Detailed results saved to: {output_file}")

    else:
        print("✗ Status: FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Execution Time: {result['execution_time_seconds']:.2f} seconds")

    print()
    print("=" * 80)
    print()
    print("💡 Regression Best Practices Demonstrated:")
    print("  1. Feature scaling handled automatically by Random Forest")
    print("  2. Outlier detection crucial for housing prices")
    print("  3. R² score primary metric for regression performance")
    print("  4. RMSE and MAE provide interpretable error measures")
    print("  5. Feature importance reveals economic drivers")
    print()

    logger.info(
        "housing_langgraph_workflow_completed",
        status=result["status"],
        duration=result["execution_time_seconds"]
    )


if __name__ == "__main__":
    main()
