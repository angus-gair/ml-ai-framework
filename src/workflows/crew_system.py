"""CrewAI-based multi-agent workflow implementation."""

from typing import Any, Dict, List, Optional
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from structlog import get_logger

from ..models.schemas import AnalysisReport, TaskInput, TaskOutput, TaskStatus
from ..tools.data_tools import load_dataset, clean_data, calculate_statistics
from ..tools.ml_tools import train_model, evaluate_model
from ..tools.analysis_tools import perform_eda

logger = get_logger(__name__)


class CrewAIWorkflow:
    """CrewAI-based workflow for ML pipeline."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_iterations: int = 10,
    ):
        """
        Initialize CrewAI workflow.

        Args:
            model_name: LLM model to use
            temperature: LLM temperature
            max_iterations: Maximum agent iterations
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.max_iterations = max_iterations
        logger.info("crewai_workflow_initialized", model=model_name)

    def create_agents(self) -> List[Agent]:
        """Create specialized agents for the workflow."""
        agents = []

        # 1. Data Loading Agent
        data_loader = Agent(
            role="Data Loading Specialist",
            goal="Load and validate datasets efficiently",
            backstory=(
                "Expert in data ingestion with deep knowledge of various file formats. "
                "Ensures data integrity and proper validation during loading."
            ),
            llm=self.llm,
            tools=[load_dataset, calculate_statistics],
            verbose=True,
            max_iter=self.max_iterations,
        )
        agents.append(data_loader)

        # 2. Data Cleaning Agent
        data_cleaner = Agent(
            role="Data Cleaning Specialist",
            goal="Clean and prepare data for analysis",
            backstory=(
                "Meticulous data scientist specializing in data quality. "
                "Expert in handling missing values, outliers, and data inconsistencies."
            ),
            llm=self.llm,
            tools=[clean_data, calculate_statistics],
            verbose=True,
            max_iter=self.max_iterations,
        )
        agents.append(data_cleaner)

        # 3. EDA Agent
        eda_specialist = Agent(
            role="Exploratory Data Analysis Expert",
            goal="Uncover insights through comprehensive statistical analysis",
            backstory=(
                "Statistical analyst with expertise in pattern recognition. "
                "Skilled in identifying relationships, distributions, and anomalies."
            ),
            llm=self.llm,
            tools=[perform_eda, calculate_statistics],
            verbose=True,
            max_iter=self.max_iterations,
        )
        agents.append(eda_specialist)

        # 4. Model Training Agent
        model_trainer = Agent(
            role="Machine Learning Engineer",
            goal="Train optimal models with best practices",
            backstory=(
                "Senior ML engineer with expertise in model selection and hyperparameter tuning. "
                "Focuses on creating robust, generalizable models."
            ),
            llm=self.llm,
            tools=[train_model, evaluate_model],
            verbose=True,
            max_iter=self.max_iterations,
        )
        agents.append(model_trainer)

        # 5. Report Generation Agent
        report_generator = Agent(
            role="ML Report Analyst",
            goal="Create comprehensive analysis reports with actionable insights",
            backstory=(
                "Technical writer and data scientist who excels at translating complex "
                "ML findings into clear, actionable business insights."
            ),
            llm=self.llm,
            tools=[],
            verbose=True,
            max_iter=self.max_iterations,
        )
        agents.append(report_generator)

        logger.info("agents_created", count=len(agents))
        return agents

    def create_tasks(
        self,
        agents: List[Agent],
        data_path: str,
        target_column: Optional[str] = None,
    ) -> List[Task]:
        """
        Create tasks for the workflow.

        Args:
            agents: List of agents
            data_path: Path to dataset
            target_column: Target variable name

        Returns:
            List of tasks
        """
        tasks = []

        # Task 1: Load Data
        load_task = Task(
            description=f"Load dataset from {data_path} and generate metadata statistics.",
            agent=agents[0],
            expected_output="Dataset loaded with metadata and initial statistics.",
        )
        tasks.append(load_task)

        # Task 2: Clean Data
        clean_task = Task(
            description=(
                "Clean the loaded dataset by handling missing values, removing duplicates, "
                "and detecting outliers. Provide detailed cleaning report."
            ),
            agent=agents[1],
            expected_output="Cleaned dataset with comprehensive cleaning report.",
            context=[load_task],
        )
        tasks.append(clean_task)

        # Task 3: Perform EDA
        eda_task = Task(
            description=(
                f"Perform comprehensive exploratory data analysis. "
                f"{'Focus on target variable: ' + target_column if target_column else ''}"
                "Analyze distributions, correlations, and patterns."
            ),
            agent=agents[2],
            expected_output="Detailed EDA findings with statistical insights and visualizations plan.",
            context=[clean_task],
        )
        tasks.append(eda_task)

        # Task 4: Train Model
        train_task = Task(
            description=(
                f"Train machine learning model{f' to predict {target_column}' if target_column else ''}. "
                "Perform cross-validation and hyperparameter tuning for optimal performance."
            ),
            agent=agents[3],
            expected_output="Trained model with performance metrics and evaluation report.",
            context=[eda_task],
        )
        tasks.append(train_task)

        # Task 5: Generate Report
        report_task = Task(
            description=(
                "Synthesize all findings into a comprehensive analysis report. "
                "Include dataset metadata, cleaning summary, EDA insights, model performance, "
                "and actionable recommendations."
            ),
            agent=agents[4],
            expected_output="Complete analysis report with insights and recommendations.",
            context=[load_task, clean_task, eda_task, train_task],
        )
        tasks.append(report_task)

        logger.info("tasks_created", count=len(tasks))
        return tasks

    def run(
        self,
        data_path: str,
        target_column: Optional[str] = None,
        process_type: Process = Process.sequential,
    ) -> Dict[str, Any]:
        """
        Execute the CrewAI workflow.

        Args:
            data_path: Path to dataset
            target_column: Target variable name
            process_type: Execution process (sequential or hierarchical)

        Returns:
            Workflow results
        """
        start_time = datetime.utcnow()
        logger.info("crewai_workflow_starting", data_path=data_path, target=target_column)

        try:
            # Create agents and tasks
            agents = self.create_agents()
            tasks = self.create_tasks(agents, data_path, target_column)

            # Create and run crew
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=process_type,
                verbose=True,
            )

            result = crew.kickoff()

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            logger.info(
                "crewai_workflow_completed",
                duration_seconds=round(execution_time, 2),
            )

            return {
                "status": "success",
                "result": result,
                "execution_time_seconds": round(execution_time, 2),
                "agents_used": len(agents),
                "tasks_completed": len(tasks),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                "crewai_workflow_failed",
                error=str(e),
                duration_seconds=round(execution_time, 2),
            )

            return {
                "status": "failed",
                "error": str(e),
                "execution_time_seconds": round(execution_time, 2),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def run_async(
        self,
        data_path: str,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute workflow asynchronously.

        Args:
            data_path: Path to dataset
            target_column: Target variable name

        Returns:
            Workflow results
        """
        logger.info("crewai_async_workflow_starting", data_path=data_path)

        # For async execution, use hierarchical process
        return self.run(data_path, target_column, process_type=Process.hierarchical)
