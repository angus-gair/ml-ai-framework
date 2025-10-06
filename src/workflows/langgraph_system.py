"""LangGraph-based workflow implementation with state management."""

from typing import Any, Dict, List, TypedDict, Annotated, Sequence
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from structlog import get_logger

from ..models.schemas import TaskStatus
from ..tools.data_tools import load_dataset, clean_data, calculate_statistics
from ..tools.ml_tools import train_model, evaluate_model
from ..tools.analysis_tools import perform_eda

logger = get_logger(__name__)


class AgentState(TypedDict):
    """State for LangGraph workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data_path: str
    target_column: str
    dataset: Any
    dataset_metadata: Any
    cleaned_data: Any
    cleaning_report: Any
    eda_findings: Any
    trained_model: Any
    model_performance: Any
    final_report: Dict[str, Any]
    current_step: str
    errors: List[str]


class LangGraphWorkflow:
    """LangGraph-based workflow for ML pipeline."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
    ):
        """
        Initialize LangGraph workflow.

        Args:
            model_name: LLM model to use
            temperature: LLM temperature
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.workflow = self._build_workflow()
        logger.info("langgraph_workflow_initialized", model=model_name)

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("load_data", self._load_data_node)
        workflow.add_node("clean_data", self._clean_data_node)
        workflow.add_node("perform_eda", self._eda_node)
        workflow.add_node("train_model", self._train_model_node)
        workflow.add_node("generate_report", self._report_node)

        # Define edges
        workflow.set_entry_point("load_data")
        workflow.add_edge("load_data", "clean_data")
        workflow.add_edge("clean_data", "perform_eda")
        workflow.add_edge("perform_eda", "train_model")
        workflow.add_edge("train_model", "generate_report")
        workflow.add_edge("generate_report", END)

        logger.info("workflow_graph_built", nodes=5)
        return workflow.compile()

    def _load_data_node(self, state: AgentState) -> AgentState:
        """Load data node."""
        logger.info("executing_load_data_node", path=state["data_path"])

        try:
            df, metadata = load_dataset(state["data_path"])

            state["dataset"] = df
            state["dataset_metadata"] = metadata
            state["current_step"] = "load_data"
            state["messages"].append(
                AIMessage(content=f"Loaded dataset: {metadata.shape[0]} rows, {metadata.shape[1]} columns")
            )

            logger.info("load_data_node_completed", rows=metadata.shape[0])

        except Exception as e:
            error_msg = f"Data loading failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
            logger.error("load_data_node_failed", error=str(e))

        return state

    def _clean_data_node(self, state: AgentState) -> AgentState:
        """Clean data node."""
        logger.info("executing_clean_data_node")

        try:
            df = state["dataset"]
            cleaned_df, report = clean_data(
                df,
                drop_duplicates=True,
                handle_missing="fill_mean",
                missing_threshold=0.5,
                outlier_method="iqr",
            )

            state["cleaned_data"] = cleaned_df
            state["cleaning_report"] = report
            state["current_step"] = "clean_data"
            state["messages"].append(
                AIMessage(
                    content=f"Cleaned dataset: {report.rows_removed} rows removed, "
                    f"{len(report.operations)} operations performed"
                )
            )

            logger.info("clean_data_node_completed", rows_removed=report.rows_removed)

        except Exception as e:
            error_msg = f"Data cleaning failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
            logger.error("clean_data_node_failed", error=str(e))

        return state

    def _eda_node(self, state: AgentState) -> AgentState:
        """EDA node."""
        logger.info("executing_eda_node")

        try:
            df = state["cleaned_data"]
            target = state.get("target_column")

            findings = perform_eda(df, target_column=target)

            state["eda_findings"] = findings
            state["current_step"] = "perform_eda"
            state["messages"].append(
                AIMessage(
                    content=f"EDA completed: {len(findings.statistical_summary)} features analyzed, "
                    f"{findings.total_outliers} outliers detected"
                )
            )

            logger.info("eda_node_completed", outliers=findings.total_outliers)

        except Exception as e:
            error_msg = f"EDA failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
            logger.error("eda_node_failed", error=str(e))

        return state

    def _train_model_node(self, state: AgentState) -> AgentState:
        """Model training node."""
        logger.info("executing_train_model_node")

        try:
            df = state["cleaned_data"]
            target = state.get("target_column")

            if not target or target not in df.columns:
                raise ValueError("Target column not specified or not found in dataset")

            # Prepare features and target
            X = df.drop(columns=[target])
            y = df[target]

            # Determine model type
            if y.dtype == 'object' or y.nunique() < 20:
                model_type = "classification"
            else:
                model_type = "regression"

            # Train model
            model, performance = train_model(
                X, y,
                model_type=model_type,
                algorithm="random_forest",
                hyperparameters={"n_estimators": 100, "random_state": 42},
            )

            state["trained_model"] = model
            state["model_performance"] = performance
            state["current_step"] = "train_model"

            primary_metric = performance.validation_metrics.accuracy or performance.validation_metrics.r2_score
            state["messages"].append(
                AIMessage(
                    content=f"Model trained: {model_type} with validation score {primary_metric:.4f}"
                )
            )

            logger.info("train_model_node_completed", score=primary_metric)

        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
            logger.error("train_model_node_failed", error=str(e))

        return state

    def _report_node(self, state: AgentState) -> AgentState:
        """Report generation node."""
        logger.info("executing_report_node")

        try:
            # Compile final report
            report = {
                "status": "success" if not state["errors"] else "completed_with_errors",
                "dataset_metadata": {
                    "name": state["dataset_metadata"].name,
                    "shape": state["dataset_metadata"].shape,
                    "memory_mb": state["dataset_metadata"].total_memory_mb,
                },
                "cleaning_summary": {
                    "rows_removed": state["cleaning_report"].rows_removed,
                    "operations_count": len(state["cleaning_report"].operations),
                },
                "eda_summary": {
                    "features_analyzed": len(state["eda_findings"].statistical_summary),
                    "outliers_detected": state["eda_findings"].total_outliers,
                    "high_correlations": len(state["eda_findings"].correlations),
                },
                "model_performance": {
                    "model_type": state["model_performance"].model_type.value,
                    "algorithm": state["model_performance"].model_name,
                    "validation_score": (
                        state["model_performance"].validation_metrics.accuracy or
                        state["model_performance"].validation_metrics.r2_score
                    ),
                    "training_time": state["model_performance"].training_duration_seconds,
                },
                "errors": state["errors"],
                "timestamp": datetime.utcnow().isoformat(),
            }

            state["final_report"] = report
            state["current_step"] = "generate_report"
            state["messages"].append(
                AIMessage(content="Analysis report generated successfully")
            )

            logger.info("report_node_completed", status=report["status"])

        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
            logger.error("report_node_failed", error=str(e))

        return state

    def run(
        self,
        data_path: str,
        target_column: str,
    ) -> Dict[str, Any]:
        """
        Execute the LangGraph workflow.

        Args:
            data_path: Path to dataset
            target_column: Target variable name

        Returns:
            Workflow results
        """
        start_time = datetime.utcnow()
        logger.info("langgraph_workflow_starting", data_path=data_path, target=target_column)

        try:
            # Initialize state
            initial_state = AgentState(
                messages=[HumanMessage(content=f"Analyze dataset at {data_path}")],
                data_path=data_path,
                target_column=target_column,
                dataset=None,
                dataset_metadata=None,
                cleaned_data=None,
                cleaning_report=None,
                eda_findings=None,
                trained_model=None,
                model_performance=None,
                final_report={},
                current_step="initialized",
                errors=[],
            )

            # Execute workflow
            final_state = self.workflow.invoke(initial_state)

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            result = {
                "status": final_state["final_report"].get("status", "unknown"),
                "report": final_state["final_report"],
                "execution_time_seconds": round(execution_time, 2),
                "messages": [msg.content for msg in final_state["messages"]],
                "timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                "langgraph_workflow_completed",
                status=result["status"],
                duration_seconds=round(execution_time, 2),
            )

            return result

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                "langgraph_workflow_failed",
                error=str(e),
                duration_seconds=round(execution_time, 2),
            )

            return {
                "status": "failed",
                "error": str(e),
                "execution_time_seconds": round(execution_time, 2),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def run_async(
        self,
        data_path: str,
        target_column: str,
    ) -> Dict[str, Any]:
        """
        Execute workflow asynchronously.

        Args:
            data_path: Path to dataset
            target_column: Target variable name

        Returns:
            Workflow results
        """
        logger.info("langgraph_async_workflow_starting", data_path=data_path)

        # LangGraph supports async execution natively
        # For now, using sync version
        return self.run(data_path, target_column)
