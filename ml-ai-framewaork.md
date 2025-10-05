# Multi-Agent AI System Technical Blueprint: Production-Grade Data Analysis Platform

**This blueprint delivers a complete architecture for building autonomous, type-safe multi-agent systems that perform end-to-end data analysis—from acquisition through predictive modeling to reporting.** The system uses specialized AI agents coordinated through either CrewAI or LangGraph, with strict Pydantic validation ensuring data integrity and AG-UI protocol enabling real-time frontend communication. You'll find production-ready code patterns, framework comparisons, and battle-tested practices for error handling, testing, and deployment.

Built on modern Python frameworks (CrewAI 0.70+, LangGraph 2024-2025, Pydantic v2), this architecture supports sequential and hierarchical workflows, implements comprehensive observability, and scales from prototype to production. The five specialized agents—DataAcquisition, DataCleaning, ExploratoryAnalysis, PredictiveModeling, and Reporting—collaborate through well-defined interfaces, with each agent's inputs and outputs validated by Pydantic schemas. Whether you're building a research platform, business intelligence system, or ML operations pipeline, this blueprint provides the technical foundation and implementation patterns you need.

## System architecture: Five specialized agents working in concert

The multi-agent system employs a **division-of-labor architecture** where each agent masters a specific domain. The DataAcquisitionAgent handles dataset discovery and loading from public repositories, APIs, or file systems. It validates data schemas and ensures completeness before passing sanitized data to downstream agents. The DataCleaningAgent then performs preprocessing—handling missing values through imputation strategies, removing duplicates, normalizing features, and detecting outliers using statistical methods like IQR or Z-score analysis.

Next, the ExploratoryAnalysisAgent generates comprehensive statistical summaries, correlation matrices, and distribution analyses. It identifies patterns through univariate and bivariate analysis, flags anomalies, and recommends features for modeling based on variance, correlation strength, and domain relevance. The PredictiveModelingAgent selects appropriate algorithms (classification for categorical targets, regression for continuous outcomes), performs train-test splits, executes hyperparameter tuning via grid search or Bayesian optimization, and evaluates performance using metrics like accuracy, precision, recall, F1-score, or R² depending on the problem type.

Finally, the ReportingAgent synthesizes findings from all previous agents into a **human-readable executive report**. It translates statistical findings into business insights, includes visualizations (referenced as descriptions since agents don't generate images directly), provides model interpretation through feature importance, and delivers actionable recommendations. The agent outputs conform to AG-UI protocol for real-time streaming to frontend applications.

Each agent maintains **clear boundaries** through Pydantic-validated interfaces. Agents never directly access each other's internal state; instead, they communicate through typed messages containing only necessary context. This loose coupling enables independent testing, parallel development, and easy agent replacement or upgrade without cascading changes.

## Framework selection: CrewAI versus LangGraph for different use cases

**CrewAI excels at rapid prototyping and straightforward sequential workflows.** Its high-level abstraction uses intuitive role/goal/backstory definitions that map naturally to team structures. You define agents declaratively in YAML, write minimal orchestration code, and leverage built-in patterns for delegation and hierarchical management. CrewAI version 0.70+ offers native Pydantic integration through `output_pydantic` parameters, automatic context passing between tasks, and three memory types (short-term, long-term, entity) with minimal configuration.

**LangGraph provides superior control for complex, non-linear workflows** requiring conditional logic, dynamic routing, or sophisticated state management. Built on graph primitives (nodes, edges, state), it enables explicit workflow definition with full visibility into execution paths. LangGraph's checkpointing system provides automatic state persistence, crash recovery, and human-in-the-loop (HITL) capabilities through interrupt points. Its integration with LangSmith delivers production-grade observability—tracing every agent decision, tool call, and state transition.

**Choose CrewAI when:** Your workflow is primarily sequential or hierarchical, you value rapid development over fine-grained control, your team prefers declarative configuration, and you don't need custom routing logic. CrewAI's opinionated design accelerates development for 80% of use cases—research pipelines, content generation workflows, and business process automation.

**Choose LangGraph when:** You need dynamic agent selection based on runtime conditions, require durable execution with guaranteed state consistency, want explicit control over every workflow transition, or are building production systems demanding audit trails and debugging capabilities. LangGraph suits complex scenarios like multi-step reasoning, adaptive workflows, and systems requiring regulatory compliance documentation.

For our data analysis system, **both frameworks work well**. CrewAI offers faster initial development, while LangGraph provides more flexibility for handling edge cases like partial data, model training failures, or iterative refinement loops. The code examples below demonstrate both approaches, allowing you to choose based on your team's needs.

## Implementation guide: Step-by-step system construction

### Step 1: Environment setup and dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install core dependencies
pip install crewai crewai-tools  # For CrewAI approach
pip install langgraph langchain-openai langchain-community  # For LangGraph
pip install pydantic pydantic-ai-slim  # Type safety
pip install pandas numpy scikit-learn matplotlib seaborn  # Data tools
pip install structlog pytest pytest-asyncio  # Logging and testing

# Optional: AG-UI protocol
pip install 'pydantic-ai-slim[ag-ui]'
```

Create `.env` file for API keys:
```bash
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here  # Optional alternative
```

### Step 2: Define Pydantic models for type safety

These schemas enforce strict validation at every agent boundary:

```python
# models.py - Comprehensive type-safe data models
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Optional, Literal
from datetime import datetime
import uuid

class DatasetMetadata(BaseModel):
    """Schema for dataset validation and tracking."""
    model_config = ConfigDict(strict=True)
    
    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(min_length=1)
    source: str
    rows: int = Field(gt=0)
    columns: int = Field(gt=0)
    dtypes: Dict[str, str]
    missing_values: Dict[str, int] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class CleaningReport(BaseModel):
    """Output from DataCleaningAgent."""
    model_config = ConfigDict(strict=True)
    
    rows_before: int = Field(gt=0)
    rows_after: int = Field(gt=0)
    duplicates_removed: int = Field(ge=0)
    missing_values_handled: Dict[str, str]  # column -> strategy
    outliers_detected: Dict[str, int]
    feature_engineering: List[str] = Field(default_factory=list)
    cleaned_data_path: str
    
    @field_validator('rows_after')
    @classmethod
    def validate_rows(cls, v, info):
        if 'rows_before' in info.data and v > info.data['rows_before']:
            raise ValueError('rows_after cannot exceed rows_before')
        return v

class CorrelationPair(BaseModel):
    """Represents a correlation between two features."""
    feature_a: str
    feature_b: str
    correlation: float = Field(ge=-1.0, le=1.0)
    significance: Literal["weak", "moderate", "strong"]

class EDAFindings(BaseModel):
    """Output from ExploratoryAnalysisAgent."""
    summary_statistics: Dict[str, Dict[str, float]]
    strong_correlations: List[CorrelationPair] = Field(min_length=0)
    outliers_by_column: Dict[str, int]
    distribution_insights: List[str]
    recommended_features: List[str] = Field(min_length=1)
    visualizations_created: List[str]  # Paths or descriptions

class ModelPerformance(BaseModel):
    """Output from PredictiveModelingAgent."""
    algorithm: str
    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    feature_importance: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]] = None
    model_path: str
    training_time_seconds: float

class AnalysisReport(BaseModel):
    """Final output from ReportingAgent."""
    dataset_name: str
    data_quality: CleaningReport
    eda_findings: EDAFindings
    model_performance: ModelPerformance
    executive_summary: str = Field(min_length=100)
    key_insights: List[str] = Field(min_length=3)
    recommendations: List[str] = Field(min_length=3)
    generated_at: datetime = Field(default_factory=datetime.now)
```

### Step 3: Build custom tools for agents

Tools are the hands of your agents—functions they call to interact with the world:

```python
# tools.py - Custom tools with Pydantic validation
from crewai.tools import tool
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class DataLoadInput(BaseModel):
    file_path: str = Field(description="Path to CSV file")

@tool("Load Dataset")
def load_dataset(file_path: str) -> str:
    """Load CSV dataset and return metadata JSON string."""
    try:
        df = pd.read_csv(file_path)
        metadata = {
            "name": file_path.split('/')[-1],
            "source": file_path,
            "rows": len(df),
            "columns": len(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {col: int(df[col].isna().sum()) for col in df.columns}
        }
        return str(metadata)
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

@tool("Clean Data")
def clean_data(input_path: str, output_path: str) -> str:
    """Clean dataset: handle missing values, remove duplicates, detect outliers."""
    df = pd.read_csv(input_path)
    rows_before = len(df)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    
    # Detect outliers using IQR
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)
        outliers[col] = outlier_mask.sum()
    
    df.to_csv(output_path, index=False)
    
    return f"Cleaned {rows_before} rows to {len(df)}. Removed {duplicates} duplicates. Outliers: {outliers}"

@tool("Calculate Statistics")
def calculate_statistics(file_path: str) -> str:
    """Generate comprehensive statistical summary."""
    df = pd.read_csv(file_path)
    stats = df.describe().to_dict()
    
    # Add correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr().to_dict()
    
    return f"Statistics: {stats}\n\nCorrelations: {correlations}"

@tool("Train Model")
def train_model(data_path: str, target_column: str, model_output_path: str) -> str:
    """Train classification model and return performance metrics."""
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
    y = df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "feature_importance": dict(zip(X.columns, model.feature_importances_))
    }
    
    # Save model
    joblib.dump(model, model_output_path)
    
    return str(metrics)
```

### Step 4: CrewAI implementation

Complete multi-agent system with CrewAI's declarative approach:

```python
# crew_system.py - CrewAI implementation
from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, task, crew
from models import DatasetMetadata, CleaningReport, EDAFindings, ModelPerformance, AnalysisReport
from tools import load_dataset, clean_data, calculate_statistics, train_model
import os

os.environ["OPENAI_API_KEY"] = "your-key"

llm = LLM(model="gpt-4o", temperature=0.7)

@CrewBase
class DataAnalysisCrew:
    """Multi-agent data analysis system using CrewAI."""
    
    @agent
    def data_acquisition_agent(self) -> Agent:
        return Agent(
            role="Data Acquisition Specialist",
            goal="Load and validate datasets from various sources",
            backstory="""You're an expert in data engineering with deep knowledge of 
            data formats, APIs, and validation techniques. You ensure data quality 
            from the very first step.""",
            tools=[load_dataset],
            verbose=True,
            llm=llm,
            max_iter=10
        )
    
    @agent
    def data_cleaning_agent(self) -> Agent:
        return Agent(
            role="Data Quality Engineer",
            goal="Clean and prepare data for analysis",
            backstory="""You have 10+ years of experience in data preprocessing. 
            You identify and fix data quality issues with surgical precision.""",
            tools=[clean_data],
            verbose=True,
            llm=llm,
            max_iter=15
        )
    
    @agent
    def exploratory_analysis_agent(self) -> Agent:
        return Agent(
            role="Data Scientist - EDA Specialist",
            goal="Perform comprehensive exploratory data analysis",
            backstory="""You excel at statistical analysis and pattern recognition. 
            You uncover hidden insights that others miss.""",
            tools=[calculate_statistics],
            verbose=True,
            llm=llm,
            max_iter=15
        )
    
    @agent
    def predictive_modeling_agent(self) -> Agent:
        return Agent(
            role="Machine Learning Engineer",
            goal="Build and evaluate predictive models",
            backstory="""You're a skilled ML practitioner who selects optimal 
            algorithms and tunes models to perfection.""",
            tools=[train_model],
            verbose=True,
            llm=llm,
            max_iter=20
        )
    
    @agent
    def reporting_agent(self) -> Agent:
        return Agent(
            role="Data Storyteller",
            goal="Create comprehensive analytical reports",
            backstory="""You translate complex technical findings into clear, 
            actionable business insights.""",
            verbose=True,
            llm=llm,
            max_iter=10
        )
    
    @task
    def acquisition_task(self) -> Task:
        return Task(
            description="""Load the dataset from {data_path} and validate its structure.
            Confirm row counts, column types, and identify missing values.""",
            expected_output="JSON string with dataset metadata including row count, columns, and data types",
            agent=self.data_acquisition_agent()
        )
    
    @task
    def cleaning_task(self) -> Task:
        return Task(
            description="""Clean the dataset by:
            1. Handling missing values (median for numeric, mode for categorical)
            2. Removing duplicates
            3. Detecting outliers using IQR method
            4. Saving cleaned data to ./data/cleaned_data.csv""",
            expected_output="Cleaning report with before/after statistics",
            agent=self.data_cleaning_agent(),
            output_pydantic=CleaningReport,
            context=[self.acquisition_task()]
        )
    
    @task
    def eda_task(self) -> Task:
        return Task(
            description="""Perform exploratory data analysis on cleaned data:
            1. Calculate descriptive statistics
            2. Identify strong correlations (|r| > 0.5)
            3. Analyze distributions
            4. Recommend features for modeling based on variance and correlation""",
            expected_output="EDA findings with statistics, correlations, and feature recommendations",
            agent=self.exploratory_analysis_agent(),
            output_pydantic=EDAFindings,
            context=[self.cleaning_task()],
            markdown=True,
            output_file="eda_report.md"
        )
    
    @task
    def modeling_task(self) -> Task:
        return Task(
            description="""Build a predictive model for {target_variable}:
            1. Select appropriate algorithm (Random Forest for this task)
            2. Train model on recommended features
            3. Evaluate using accuracy, precision, recall, F1-score
            4. Extract feature importance
            5. Save model to ./models/trained_model.pkl""",
            expected_output="Model performance metrics and feature importance",
            agent=self.predictive_modeling_agent(),
            output_pydantic=ModelPerformance,
            context=[self.eda_task()]
        )
    
    @task
    def reporting_task(self) -> Task:
        return Task(
            description="""Synthesize all findings into an executive report:
            1. Summarize data quality improvements
            2. Highlight key EDA insights
            3. Explain model performance
            4. Provide 3+ actionable business recommendations
            5. Include executive summary (min 100 words)""",
            expected_output="Comprehensive markdown report suitable for stakeholders",
            agent=self.reporting_agent(),
            output_pydantic=AnalysisReport,
            context=[self.cleaning_task(), self.eda_task(), self.modeling_task()],
            markdown=True,
            output_file="final_report.md"
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            embedder={
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"}
            }
        )

# Execute the crew
if __name__ == "__main__":
    inputs = {
        'data_path': 'data/sample_dataset.csv',
        'target_variable': 'target'
    }
    
    result = DataAnalysisCrew().crew().kickoff(inputs=inputs)
    
    # Access structured outputs
    print("\n=== CLEANING REPORT ===")
    print(result.tasks[1].output.pydantic)
    
    print("\n=== MODEL PERFORMANCE ===")
    print(result.tasks[3].output.pydantic.model_dump())
```

### Step 5: LangGraph implementation

Graph-based approach with explicit control flow:

```python
# langgraph_system.py - LangGraph implementation
from typing import Literal, Annotated, TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from models import CleaningReport, EDAFindings, ModelPerformance
import operator

# Define state
class AnalysisState(TypedDict):
    messages: Annotated[list, operator.add]
    dataset_metadata: dict
    cleaning_report: CleaningReport | None
    eda_findings: EDAFindings | None
    model_performance: ModelPerformance | None
    current_stage: str
    data_path: str
    target_variable: str

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Supervisor node
def supervisor(state: AnalysisState) -> Command[Literal["data_acquisition", "data_cleaning", "eda", "modeling", "reporting", "__end__"]]:
    """Route to appropriate agent based on workflow stage."""
    
    class Router(TypedDict):
        reasoning: str
        next: Literal["data_acquisition", "data_cleaning", "eda", "modeling", "reporting", "FINISH"]
    
    system_prompt = f"""You are orchestrating a data analysis workflow.
    Current stage: {state['current_stage']}
    
    Workflow order:
    1. data_acquisition -> load dataset
    2. data_cleaning -> clean and preprocess
    3. eda -> exploratory analysis
    4. modeling -> train predictive model
    5. reporting -> generate final report
    6. FINISH
    
    Determine the next agent to invoke or FINISH if complete."""
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    
    goto = END if response["next"] == "FINISH" else response["next"]
    
    return Command(
        goto=goto,
        update={"current_stage": response["next"]}
    )

# Agent nodes
def data_acquisition_node(state: AnalysisState) -> Command[Literal["supervisor"]]:
    """Load and validate dataset."""
    result = load_dataset(state["data_path"])
    
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=f"Dataset loaded: {result}", name="data_acquisition")],
            "dataset_metadata": eval(result) if not result.startswith("Error") else {}
        }
    )

def data_cleaning_node(state: AnalysisState) -> Command[Literal["supervisor"]]:
    """Clean dataset."""
    result = clean_data(state["data_path"], "./data/cleaned_data.csv")
    
    # Parse result into CleaningReport (simplified)
    cleaning_report = CleaningReport(
        rows_before=state["dataset_metadata"].get("rows", 0),
        rows_after=state["dataset_metadata"].get("rows", 0),
        duplicates_removed=0,
        missing_values_handled={},
        outliers_detected={},
        cleaned_data_path="./data/cleaned_data.csv"
    )
    
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=f"Cleaning complete: {result}", name="data_cleaning")],
            "cleaning_report": cleaning_report
        }
    )

def eda_node(state: AnalysisState) -> Command[Literal["supervisor"]]:
    """Perform exploratory analysis."""
    result = calculate_statistics("./data/cleaned_data.csv")
    
    eda_findings = EDAFindings(
        summary_statistics={},
        strong_correlations=[],
        outliers_by_column={},
        distribution_insights=["Analysis complete"],
        recommended_features=["feature1", "feature2"],
        visualizations_created=[]
    )
    
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=f"EDA complete: {result}", name="eda_agent")],
            "eda_findings": eda_findings
        }
    )

def modeling_node(state: AnalysisState) -> Command[Literal["supervisor"]]:
    """Train predictive model."""
    result = train_model("./data/cleaned_data.csv", state["target_variable"], "./models/model.pkl")
    
    metrics = eval(result)
    model_performance = ModelPerformance(
        algorithm="RandomForest",
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        feature_importance=metrics["feature_importance"],
        model_path="./models/model.pkl",
        training_time_seconds=10.0
    )
    
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=f"Model trained: {result}", name="ml_agent")],
            "model_performance": model_performance
        }
    )

def reporting_node(state: AnalysisState) -> Command[Literal["supervisor"]]:
    """Generate final report."""
    report_content = f"""
    # Data Analysis Report
    
    Dataset: {state['dataset_metadata'].get('name', 'Unknown')}
    Model Accuracy: {state['model_performance'].accuracy:.2%}
    
    ## Key Findings
    - Data cleaned successfully
    - Model trained with {state['model_performance'].f1_score:.2%} F1-score
    - Recommended features identified
    """
    
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=f"Report generated:\n{report_content}", name="reporter")]
        }
    )

# Build graph
builder = StateGraph(AnalysisState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor)
builder.add_node("data_acquisition", data_acquisition_node)
builder.add_node("data_cleaning", data_cleaning_node)
builder.add_node("eda", eda_node)
builder.add_node("modeling", modeling_node)
builder.add_node("reporting", reporting_node)

# Compile with checkpointing
checkpointer = SqliteSaver.from_conn_string("analysis_checkpoints.db")
graph = builder.compile(checkpointer=checkpointer)

# Execute
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "analysis-001"}}
    
    initial_state = {
        "messages": [HumanMessage(content="Start data analysis")],
        "dataset_metadata": {},
        "cleaning_report": None,
        "eda_findings": None,
        "model_performance": None,
        "current_stage": "start",
        "data_path": "data/sample_dataset.csv",
        "target_variable": "target"
    }
    
    for step in graph.stream(initial_state, config, stream_mode="values"):
        if step["messages"]:
            print(f"\n{step['messages'][-1].name}: {step['messages'][-1].content}")
```

### Step 6: AG-UI protocol integration

Enable real-time streaming to frontend applications:

```python
# ag_ui_server.py - FastAPI server with AG-UI protocol
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent
from pydantic_ai.ag_ui import StateDeps
from ag_ui.core import RunAgentInput, EventType, TextMessageContentEvent, RunStartedEvent, RunFinishedEvent
from ag_ui.encoder import EventEncoder
from models import AnalysisReport
import uuid

app = FastAPI()

# Create Pydantic AI agent
analysis_agent = Agent(
    'openai:gpt-4o',
    deps_type=StateDeps[AnalysisReport],
    output_type=AnalysisReport,
    system_prompt="You are a data analysis expert that produces structured reports."
)

@app.post('/agent')
async def run_analysis_agent(input_data: RunAgentInput):
    """Stream analysis results using AG-UI protocol."""
    
    async def event_generator():
        encoder = EventEncoder()
        
        # Send run started
        yield encoder.encode(
            RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id
            )
        )
        
        # Process with agent (simplified - would integrate with CrewAI/LangGraph)
        message_id = uuid.uuid4()
        
        # Stream message content
        response_parts = [
            "Analyzing dataset...",
            "\nCleaning data...",
            "\nPerforming EDA...",
            "\nTraining model...",
            "\nGenerating report..."
        ]
        
        for part in response_parts:
            yield encoder.encode(
                TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id,
                    delta=part
                )
            )
        
        # Send run finished
        yield encoder.encode(
            RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id
            )
        )
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# Or use Pydantic AI's built-in AG-UI conversion
@app.post('/agent-auto')
async def auto_ag_ui():
    """Automatic AG-UI conversion with Pydantic AI."""
    ag_ui_app = analysis_agent.to_ag_ui()
    return ag_ui_app
```

## Error handling and resilience: Production-grade fault tolerance

Multi-agent systems face unique challenges—API rate limits, network failures, invalid outputs, infinite loops. **Implement retry logic with exponential backoff** for transient failures. Wrap LLM calls in decorators that retry 3 times with increasing delays (1s, 2s, 4s) before propagating errors. Use the `retrying` library or implement custom logic.

**Circuit breakers prevent cascading failures.** When an agent consistently fails (e.g., 5 failures in 60 seconds), open the circuit—temporarily blocking requests and returning cached results or error messages. After a timeout, enter "half-open" state to test recovery. This pattern protects downstream services from overload.

**Implement graceful degradation through fallback models.** If GPT-4 hits rate limits, automatically fall back to GPT-4o-mini or Claude. Maintain a hierarchy of models ordered by capability and cost. Log every fallback event for monitoring.

**Checkpointing enables crash recovery.** LangGraph's built-in checkpointing saves state after each node execution. If a workflow crashes during modeling, restart from the last checkpoint (post-EDA) rather than reprocessing data. CrewAI requires manual checkpoint implementation—save intermediate results to disk or database after each task completion.

```python
# Error handling patterns
from retrying import retry
import logging

logger = logging.getLogger(__name__)

@retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000)
def call_llm_with_retry(prompt):
    try:
        return llm.generate(prompt)
    except RateLimitError:
        logger.warning("Rate limit hit, retrying...")
        raise

class ResilientAgent:
    def __init__(self):
        self.primary_llm = ChatOpenAI(model="gpt-4o")
        self.fallback_llm = ChatOpenAI(model="gpt-4o-mini")
    
    def generate(self, prompt):
        try:
            return call_llm_with_retry(prompt)
        except Exception as e:
            logger.error(f"Primary model failed: {e}")
            return self.fallback_llm.generate(prompt)
```

## Logging and observability: Debugging distributed agent systems

**Use structured logging with `structlog`** to emit JSON-formatted logs containing agent_id, task_id, timestamps, and custom fields. Structured logs integrate seamlessly with Elasticsearch, Splunk, or CloudWatch for querying and visualization. Every LLM call should log: model name, token count, latency, cost estimate, input/output lengths.

**Implement correlation IDs** to trace requests across agent boundaries. Generate a UUID when the workflow starts, bind it to the logger context, and include it in every log message. This enables reconstructing full execution paths from distributed logs.

**Instrument with OpenTelemetry** for distributed tracing. Create spans for each agent invocation, tool call, and LLM request. Parent-child span relationships visualize the execution graph. Integrate with Jaeger or Honeycomb for trace inspection. Critical metrics: agent latency (p50, p95, p99), token usage per agent, error rates, cache hit ratios.

```python
# Structured logging setup
import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

class ObservableAgent:
    def __init__(self, agent_id):
        self.logger = logger.bind(agent_id=agent_id)
    
    def process(self, task):
        self.logger.info("task_started", task_id=task.id)
        start = time.time()
        
        try:
            result = self._execute(task)
            latency = time.time() - start
            self.logger.info(
                "task_completed",
                latency_ms=latency * 1000,
                tokens=result.tokens
            )
            return result
        except Exception as e:
            self.logger.error("task_failed", error=str(e))
            raise
```

## Testing strategies: Validating multi-agent behavior

**Mock LLM responses for fast, deterministic unit tests.** Use `unittest.mock` or pytest fixtures to replace LLM calls with predefined outputs. Test agent logic without incurring API costs or dealing with non-deterministic responses. Pydantic AI provides `TestModel` specifically for this purpose.

**Implement LLM-as-judge evaluation** for integration tests. Generate responses from your agent system, then use a powerful model (GPT-4) to evaluate quality on dimensions like accuracy, completeness, relevance. Define clear rubrics (1-10 scales) and assert minimum scores.

**Use parameterized tests** with pytest's `@pytest.mark.parametrize` to test multiple scenarios efficiently. Create test datasets with known properties (missing values, outliers, clear correlations) and verify agents produce expected outputs.

**Test agent coordination** separately from individual agent logic. Verify supervisors route correctly, context passes between agents, and state updates propagate. Use state snapshots to assert intermediate values match expectations.

```python
# Testing examples
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.generate.return_value = "Mocked analysis result"
    return llm

def test_data_acquisition_agent(mock_llm):
    agent = DataAcquisitionAgent(llm=mock_llm)
    result = agent.load_dataset("test.csv")
    assert result is not None
    assert "rows" in result

@pytest.mark.parametrize("dataset,expected_cleaning", [
    ("data/missing_values.csv", {"missing_handled": True}),
    ("data/duplicates.csv", {"duplicates_removed": True}),
])
def test_cleaning_scenarios(dataset, expected_cleaning):
    agent = DataCleaningAgent()
    report = agent.clean(dataset)
    for key, value in expected_cleaning.items():
        assert getattr(report, key) == value
```

## State management: Coordinating information flow

**LangGraph's StateGraph** provides first-class state management. Define a TypedDict subclass specifying all state fields. Use `Annotated` with reducers (e.g., `operator.add` for message lists) to control how updates merge. State automatically persists with checkpointers, enabling pause/resume and time-travel debugging.

**CrewAI uses implicit context passing** through the `context` parameter on Tasks. Outputs from tasks listed in `context` become available to the current task. CrewAI also offers memory features—short-term (current execution), long-term (across sessions), and entity memory (tracked concepts).

**For shared mutable state**, prefer immutable updates. Instead of modifying state dictionaries in-place, return new dictionaries with updated values. This enables state rollback, simplifies debugging, and prevents race conditions in concurrent systems.

```python
# State management patterns
from typing import TypedDict, Annotated
import operator

class SharedState(TypedDict):
    messages: Annotated[list, operator.add]  # Append-only
    analysis_results: dict  # Replaced entirely
    agent_outputs: dict  # Keyed by agent_id

def agent_with_state(state: SharedState):
    # Never mutate state directly
    new_results = {**state["analysis_results"], "eda": {...}}
    
    return {
        "analysis_results": new_results,
        "messages": [{"role": "assistant", "content": "EDA complete"}]
    }
```

## Sample workflow execution on public dataset

Let's execute the complete workflow on the **Iris dataset** (classification task):

```python
# sample_run.py - Complete workflow execution
from crew_system import DataAnalysisCrew
from sklearn.datasets import load_iris
import pandas as pd
import os

# Prepare sample dataset
def setup_iris_dataset():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/iris_dataset.csv', index=False)
    print(f"Created Iris dataset: {len(df)} rows, {len(df.columns)} columns")

# Run analysis
setup_iris_dataset()

inputs = {
    'data_path': 'data/iris_dataset.csv',
    'target_variable': 'target'
}

crew = DataAnalysisCrew().crew()
result = crew.kickoff(inputs=inputs)

# Extract results
print("\n=== CLEANING REPORT ===")
cleaning = result.tasks[1].output.pydantic
print(f"Rows: {cleaning.rows_before} → {cleaning.rows_after}")
print(f"Duplicates removed: {cleaning.duplicates_removed}")

print("\n=== EDA FINDINGS ===")
eda = result.tasks[2].output.pydantic
print(f"Recommended features: {eda.recommended_features}")
print(f"Strong correlations: {len(eda.strong_correlations)}")

print("\n=== MODEL PERFORMANCE ===")
model = result.tasks[3].output.pydantic
print(f"Algorithm: {model.algorithm}")
print(f"Accuracy: {model.accuracy:.2%}")
print(f"F1 Score: {model.f1_score:.2%}")

print("\n=== FINAL REPORT ===")
report = result.tasks[4].output.pydantic
print(report.executive_summary)
print(f"\nKey insights: {report.key_insights}")
```

**Expected output:** The system loads Iris (150 rows, 5 columns), cleans data (removes any duplicates if present), performs EDA (identifies petal length/width as most important features), trains a Random Forest classifier (achieves ~95% accuracy), and generates a markdown report summarizing the classification task results.

## Framework comparison: Detailed decision matrix

| **Criterion** | **CrewAI** | **LangGraph** |
|---------------|-----------|--------------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ Declarative YAML config | ⭐⭐⭐ Explicit graph definition |
| **Flexibility** | ⭐⭐⭐ Pre-built patterns | ⭐⭐⭐⭐⭐ Full customization |
| **State Management** | ⭐⭐⭐ Context + memory | ⭐⭐⭐⭐⭐ Built-in checkpointing |
| **Production Features** | ⭐⭐⭐ Growing | ⭐⭐⭐⭐⭐ Durable, HITL, streaming |
| **Observability** | ⭐⭐⭐ Built-in logging | ⭐⭐⭐⭐⭐ LangSmith integration |
| **Learning Curve** | ⭐⭐⭐⭐⭐ Gentle | ⭐⭐⭐ Steeper |
| **Code Volume** | ⭐⭐⭐⭐⭐ Minimal | ⭐⭐⭐ More verbose |
| **Ecosystem** | ⭐⭐⭐ Growing tools | ⭐⭐⭐⭐⭐ Full LangChain access |
| **Best For** | Prototypes, linear workflows | Production, complex routing |

**CrewAI advantages:** Faster time-to-first-prototype, intuitive role-based mental model, less boilerplate for standard patterns, growing community and examples.

**LangGraph advantages:** Explicit control over routing logic, sophisticated state management with automatic persistence, streaming and HITL support out-of-the-box, comprehensive debugging through LangSmith traces, better suited for conditional workflows and dynamic agent selection.

**Recommendation:** Start with CrewAI for proof-of-concept. Migrate to LangGraph when you need production-grade durability, complex branching logic, or deep observability. Both integrate well with Pydantic for type safety.

## Key takeaways and next steps

**This blueprint provides a complete foundation** for building production-ready multi-agent data analysis systems. You've learned the architectural patterns (specialized agents with clear boundaries), implementation approaches (CrewAI for simplicity, LangGraph for control), and production practices (error handling, logging, testing, state management).

**Start by implementing the CrewAI version** with the sample Iris dataset. Verify each agent produces valid Pydantic outputs. Add custom tools for your specific data sources (SQL databases, APIs, cloud storage). Extend the EDA agent with visualization libraries (matplotlib, seaborn) to generate actual plots. Enhance the modeling agent to support multiple algorithms and hyperparameter tuning.

**Deploy with AG-UI protocol** to enable real-time frontend communication. Wrap your crew in a FastAPI server that streams events as work progresses. Build a React frontend using `@ag-ui/client` to display live status updates, intermediate results, and final reports.

**Implement comprehensive monitoring** before production deployment. Set up structured logging with correlation IDs, instrument with OpenTelemetry for distributed tracing, export metrics to Prometheus, create Grafana dashboards for latency and cost tracking, and configure alerts for error rate thresholds.

**Test thoroughly** with diverse datasets—handle edge cases like all missing values, single-column data, perfectly collinear features. Use LLM-as-judge to evaluate report quality. Implement regression tests to prevent quality degradation over time.

The future of AI systems is **collaborative intelligence**—specialized agents working in concert to solve complex problems. This blueprint gives you the technical foundation to build that future today.