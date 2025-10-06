"""Advanced Custom Agents Example.

This example demonstrates:
- Creating custom specialized agents
- Extending the framework with domain-specific tools
- Advanced agent coordination patterns
- Custom task definitions
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import Any, Dict, List
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from structlog import get_logger

from src.tools.data_tools import load_dataset, clean_data
from src.tools.ml_tools import train_model
from src.utils.logging import setup_logging

# Setup logging
setup_logging(log_level="INFO", log_to_file=True, log_dir=str(project_root / "logs"))
logger = get_logger(__name__)


class CustomMLAgents:
    """Custom agent definitions for specialized ML tasks."""

    def __init__(self, llm: ChatOpenAI):
        """Initialize with language model."""
        self.llm = llm

    def create_feature_engineer(self) -> Agent:
        """
        Create a Feature Engineering specialist.

        This agent focuses on:
        - Feature selection
        - Feature transformation
        - Dimensionality reduction
        - Feature importance analysis
        """
        return Agent(
            role="Feature Engineering Specialist",
            goal="Identify, create, and optimize features for maximum model performance",
            backstory=(
                "Expert data scientist with deep knowledge of feature engineering techniques. "
                "Specializes in creating derived features, handling high-cardinality categoricals, "
                "and applying advanced transformations like polynomial features and interactions. "
                "Known for improving model performance through intelligent feature design."
            ),
            llm=self.llm,
            tools=[],  # Add custom feature engineering tools here
            verbose=True,
            max_iter=15,
        )

    def create_hyperparameter_tuner(self) -> Agent:
        """
        Create a Hyperparameter Tuning specialist.

        This agent focuses on:
        - Grid search optimization
        - Random search strategies
        - Bayesian optimization
        - Cross-validation strategies
        """
        return Agent(
            role="Hyperparameter Optimization Expert",
            goal="Find optimal hyperparameters through systematic search strategies",
            backstory=(
                "ML engineer specializing in model optimization. Expert in various search "
                "strategies including grid search, random search, and Bayesian optimization. "
                "Uses cross-validation to ensure robust parameter selection. Known for "
                "dramatically improving model performance through careful tuning."
            ),
            llm=self.llm,
            tools=[train_model],
            verbose=True,
            max_iter=20,
        )

    def create_model_validator(self) -> Agent:
        """
        Create a Model Validation specialist.

        This agent focuses on:
        - Cross-validation strategies
        - Bias-variance analysis
        - Overfitting detection
        - Performance benchmarking
        """
        return Agent(
            role="Model Validation Specialist",
            goal="Rigorously validate models to ensure generalization and detect issues",
            backstory=(
                "Senior data scientist focused on model reliability and validation. "
                "Expert in detecting overfitting, analyzing bias-variance tradeoff, "
                "and implementing robust validation strategies. Ensures models perform "
                "well on unseen data through comprehensive testing protocols."
            ),
            llm=self.llm,
            tools=[train_model],
            verbose=True,
            max_iter=15,
        )

    def create_ensemble_architect(self) -> Agent:
        """
        Create an Ensemble Methods specialist.

        This agent focuses on:
        - Stacking models
        - Boosting techniques
        - Bagging strategies
        - Ensemble optimization
        """
        return Agent(
            role="Ensemble Methods Architect",
            goal="Design and implement ensemble models for superior predictive performance",
            backstory=(
                "ML architect specializing in ensemble methods. Expert in combining multiple "
                "models through stacking, boosting, and bagging to achieve state-of-the-art "
                "performance. Understands model diversity, voting strategies, and how to "
                "leverage strengths of different algorithms. Known for winning competitions "
                "through clever ensemble design."
            ),
            llm=self.llm,
            tools=[train_model],
            verbose=True,
            max_iter=20,
        )

    def create_data_quality_auditor(self) -> Agent:
        """
        Create a Data Quality specialist.

        This agent focuses on:
        - Data integrity checks
        - Anomaly detection
        - Distribution analysis
        - Quality reporting
        """
        return Agent(
            role="Data Quality Auditor",
            goal="Ensure data quality and identify potential issues before modeling",
            backstory=(
                "Data quality expert with rigorous standards for data validation. "
                "Specializes in detecting anomalies, identifying data drift, and ensuring "
                "data integrity. Uses statistical methods to validate distributions and "
                "flag suspicious patterns. Prevents 'garbage in, garbage out' scenarios "
                "through meticulous quality checks."
            ),
            llm=self.llm,
            tools=[load_dataset, clean_data],
            verbose=True,
            max_iter=12,
        )


def create_advanced_workflow(
    data_path: str,
    target_column: str,
    model_name: str = "gpt-4",
) -> Dict[str, Any]:
    """
    Run an advanced workflow with custom specialized agents.

    This workflow demonstrates:
    1. Data quality auditing
    2. Feature engineering
    3. Hyperparameter tuning
    4. Model validation
    5. Ensemble methods

    Args:
        data_path: Path to dataset
        target_column: Target variable name
        model_name: LLM model to use

    Returns:
        Workflow results
    """
    logger.info("starting_advanced_workflow", data_path=data_path)

    # Initialize LLM
    llm = ChatOpenAI(model=model_name, temperature=0.7)

    # Create custom agents
    custom_agents = CustomMLAgents(llm)

    agents = [
        custom_agents.create_data_quality_auditor(),
        custom_agents.create_feature_engineer(),
        custom_agents.create_hyperparameter_tuner(),
        custom_agents.create_model_validator(),
        custom_agents.create_ensemble_architect(),
    ]

    # Define advanced tasks
    tasks = [
        Task(
            description=f"Load and audit data quality for {data_path}. Identify anomalies, check distributions, and report potential issues.",
            agent=agents[0],
            expected_output="Comprehensive data quality report with recommendations.",
        ),
        Task(
            description=f"Engineer features for predicting {target_column}. Create derived features, analyze feature importance, and recommend transformations.",
            agent=agents[1],
            expected_output="Optimized feature set with engineering recommendations.",
            context=[],
        ),
        Task(
            description=f"Perform hyperparameter tuning to optimize model for {target_column} prediction. Use systematic search strategies.",
            agent=agents[2],
            expected_output="Optimal hyperparameters with performance comparison.",
            context=[],
        ),
        Task(
            description="Validate model using cross-validation and bias-variance analysis. Detect overfitting and ensure generalization.",
            agent=agents[3],
            expected_output="Model validation report with performance metrics.",
            context=[],
        ),
        Task(
            description="Design and evaluate ensemble methods to improve prediction accuracy. Compare stacking, boosting, and bagging approaches.",
            agent=agents[4],
            expected_output="Ensemble model recommendations with performance gains.",
            context=[],
        ),
    ]

    # Create and run crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    print("=" * 80)
    print("Advanced ML Workflow - Custom Specialized Agents")
    print("=" * 80)
    print()
    print("Custom Agents:")
    for i, agent in enumerate(agents, 1):
        print(f"  {i}. {agent.role}")
    print()
    print("Executing advanced workflow...")
    print("-" * 80)

    result = crew.kickoff()

    logger.info("advanced_workflow_completed")

    return {
        "status": "success",
        "result": result,
        "agents_used": len(agents),
        "tasks_completed": len(tasks),
    }


def main():
    """Run the advanced custom agents example."""
    from sklearn.datasets import load_breast_cancer
    import pandas as pd

    # Prepare breast cancer dataset
    print("Preparing breast cancer dataset for advanced analysis...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    dataset_path = "/tmp/breast_cancer_advanced.csv"
    df.to_csv(dataset_path, index=False)

    print(f"Dataset: {dataset_path}")
    print(f"Shape: {df.shape}")
    print(f"Features: {len(df.columns) - 1}")
    print()

    # Run advanced workflow
    result = create_advanced_workflow(
        data_path=dataset_path,
        target_column="target",
        model_name="gpt-4",
    )

    print()
    print("=" * 80)
    print("Advanced Workflow Results")
    print("=" * 80)
    print()
    print(f"Status: {result['status']}")
    print(f"Agents: {result['agents_used']}")
    print(f"Tasks: {result['tasks_completed']}")
    print()
    print("Final Report:")
    print("-" * 80)
    print(result['result'])
    print()

    # Save results
    output_file = project_root / "examples" / "advanced_workflow" / "custom_agents_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("Advanced Custom Agents Workflow Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(result['result']))

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
