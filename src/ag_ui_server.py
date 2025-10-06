"""AG-UI FastAPI server for streaming multi-agent workflows."""

import asyncio
import json
from typing import AsyncGenerator, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from structlog import get_logger

from .workflows.crew_system import CrewAIWorkflow
from .workflows.langgraph_system import LangGraphWorkflow
from .utils.logging import setup_logging

# Setup logging
setup_logging(log_level="INFO", json_logs=True)
logger = get_logger(__name__)


class WorkflowRequest(BaseModel):
    """Request model for workflow execution."""
    data_path: str = Field(..., description="Path to dataset file")
    target_column: Optional[str] = Field(None, description="Target variable name")
    workflow_type: str = Field("crewai", description="Workflow type: 'crewai' or 'langgraph'")
    model_name: str = Field("gpt-4", description="LLM model name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class WorkflowResponse(BaseModel):
    """Response model for workflow execution."""
    workflow_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_seconds: float
    timestamp: str


class StreamEvent(BaseModel):
    """Streaming event model."""
    event_type: str
    workflow_id: str
    timestamp: str
    data: Dict[str, Any]


# Application state
app_state = {
    "active_workflows": {},
    "completed_workflows": {},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("ag_ui_server_starting")
    yield
    logger.info("ag_ui_server_shutting_down")


# Initialize FastAPI app
app = FastAPI(
    title="ML-AI Framework AG-UI Server",
    description="Streaming multi-agent ML workflows with AG-UI protocol",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "ML-AI Framework AG-UI Server",
        "version": "0.1.0",
        "status": "running",
        "active_workflows": len(app_state["active_workflows"]),
        "completed_workflows": len(app_state["completed_workflows"]),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/workflow/execute", response_model=WorkflowResponse)
async def execute_workflow(request: WorkflowRequest):
    """
    Execute ML workflow (non-streaming).

    Args:
        request: Workflow execution request

    Returns:
        Workflow execution results
    """
    workflow_id = f"workflow-{datetime.utcnow().timestamp()}"
    logger.info(
        "workflow_execution_requested",
        workflow_id=workflow_id,
        type=request.workflow_type,
        data_path=request.data_path,
    )

    try:
        # Initialize workflow
        if request.workflow_type == "crewai":
            workflow = CrewAIWorkflow(
                model_name=request.model_name,
                temperature=request.parameters.get("temperature", 0.7),
            )
            result = workflow.run(request.data_path, request.target_column)
        elif request.workflow_type == "langgraph":
            workflow = LangGraphWorkflow(
                model_name=request.model_name,
                temperature=request.parameters.get("temperature", 0.7),
            )
            result = workflow.run(request.data_path, request.target_column)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown workflow type: {request.workflow_type}")

        # Store result
        app_state["completed_workflows"][workflow_id] = result

        response = WorkflowResponse(
            workflow_id=workflow_id,
            status=result.get("status", "unknown"),
            result=result,
            execution_time_seconds=result.get("execution_time_seconds", 0.0),
            timestamp=datetime.utcnow().isoformat(),
        )

        logger.info(
            "workflow_execution_completed",
            workflow_id=workflow_id,
            status=response.status,
        )

        return response

    except Exception as e:
        logger.error(
            "workflow_execution_failed",
            workflow_id=workflow_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


async def workflow_stream_generator(
    workflow_id: str,
    request: WorkflowRequest,
) -> AsyncGenerator[str, None]:
    """
    Generate streaming events for workflow execution.

    Args:
        workflow_id: Unique workflow identifier
        request: Workflow request

    Yields:
        SSE formatted events
    """
    try:
        # Start event
        yield json.dumps({
            "event_type": "workflow_started",
            "workflow_id": workflow_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "workflow_type": request.workflow_type,
                "data_path": request.data_path,
                "target_column": request.target_column,
            },
        }) + "\n\n"

        await asyncio.sleep(0.1)

        # Progress events (simulated for demonstration)
        steps = [
            {"step": "load_data", "message": "Loading dataset..."},
            {"step": "clean_data", "message": "Cleaning data..."},
            {"step": "perform_eda", "message": "Performing EDA..."},
            {"step": "train_model", "message": "Training model..."},
            {"step": "generate_report", "message": "Generating report..."},
        ]

        for i, step in enumerate(steps):
            yield json.dumps({
                "event_type": "progress",
                "workflow_id": workflow_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "step": step["step"],
                    "message": step["message"],
                    "progress": (i + 1) / len(steps) * 100,
                },
            }) + "\n\n"
            await asyncio.sleep(0.5)

        # Execute workflow
        if request.workflow_type == "crewai":
            workflow = CrewAIWorkflow(model_name=request.model_name)
            result = workflow.run(request.data_path, request.target_column)
        else:
            workflow = LangGraphWorkflow(model_name=request.model_name)
            result = workflow.run(request.data_path, request.target_column)

        # Completion event
        yield json.dumps({
            "event_type": "workflow_completed",
            "workflow_id": workflow_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "status": result.get("status", "unknown"),
                "result": result,
            },
        }) + "\n\n"

        app_state["completed_workflows"][workflow_id] = result

    except Exception as e:
        logger.error("workflow_stream_error", workflow_id=workflow_id, error=str(e))
        yield json.dumps({
            "event_type": "workflow_error",
            "workflow_id": workflow_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "error": str(e),
            },
        }) + "\n\n"


@app.post("/workflow/stream")
async def stream_workflow(request: WorkflowRequest):
    """
    Execute ML workflow with streaming updates (AG-UI protocol).

    Args:
        request: Workflow execution request

    Returns:
        Streaming response with workflow progress
    """
    workflow_id = f"workflow-{datetime.utcnow().timestamp()}"
    logger.info(
        "workflow_stream_requested",
        workflow_id=workflow_id,
        type=request.workflow_type,
    )

    app_state["active_workflows"][workflow_id] = {
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
    }

    return EventSourceResponse(
        workflow_stream_generator(workflow_id, request),
        media_type="text/event-stream",
    )


@app.get("/workflow/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """
    Get workflow status and results.

    Args:
        workflow_id: Workflow identifier

    Returns:
        Workflow status and results
    """
    if workflow_id in app_state["active_workflows"]:
        return {
            "workflow_id": workflow_id,
            "status": "running",
            **app_state["active_workflows"][workflow_id],
        }
    elif workflow_id in app_state["completed_workflows"]:
        return {
            "workflow_id": workflow_id,
            "status": "completed",
            "result": app_state["completed_workflows"][workflow_id],
        }
    else:
        raise HTTPException(status_code=404, detail="Workflow not found")


@app.get("/workflows")
async def list_workflows():
    """List all workflows."""
    return {
        "active": [
            {"workflow_id": wid, **data}
            for wid, data in app_state["active_workflows"].items()
        ],
        "completed": [
            {"workflow_id": wid, "status": data.get("status")}
            for wid, data in app_state["completed_workflows"].items()
        ],
    }


@app.delete("/workflow/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """
    Delete workflow results.

    Args:
        workflow_id: Workflow identifier
    """
    if workflow_id in app_state["completed_workflows"]:
        del app_state["completed_workflows"][workflow_id]
        logger.info("workflow_deleted", workflow_id=workflow_id)
        return {"message": "Workflow deleted", "workflow_id": workflow_id}
    else:
        raise HTTPException(status_code=404, detail="Workflow not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.ag_ui_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
