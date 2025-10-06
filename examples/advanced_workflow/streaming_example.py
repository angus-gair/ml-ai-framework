"""AG-UI Streaming Example.

This example demonstrates:
- Real-time streaming with AG-UI server
- WebSocket connections for live updates
- Progress monitoring during ML workflows
- Event-driven architecture
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import json
from typing import AsyncIterator, Dict, Any
from datetime import datetime
from structlog import get_logger

from src.utils.logging import setup_logging

# Setup logging
setup_logging(log_level="INFO", log_to_file=True, log_dir=str(project_root / "logs"))
logger = get_logger(__name__)


class StreamingWorkflowSimulator:
    """
    Simulate streaming workflow execution.

    In production, this would connect to the AG-UI server and stream
    real-time updates during ML workflow execution.
    """

    def __init__(self):
        """Initialize streaming simulator."""
        self.workflow_stages = [
            ("initialize", "Initializing workflow", 2),
            ("load_data", "Loading dataset", 3),
            ("validate_data", "Validating data quality", 2),
            ("clean_data", "Cleaning data", 4),
            ("perform_eda", "Performing exploratory analysis", 5),
            ("feature_engineering", "Engineering features", 4),
            ("train_model", "Training model", 8),
            ("evaluate_model", "Evaluating performance", 3),
            ("generate_report", "Generating report", 2),
            ("finalize", "Finalizing workflow", 1),
        ]

    async def stream_workflow_progress(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream workflow progress updates.

        Yields:
            Dict containing stage information and progress
        """
        total_stages = len(self.workflow_stages)
        start_time = datetime.utcnow()

        # Initial event
        yield {
            "event": "workflow_started",
            "timestamp": start_time.isoformat(),
            "total_stages": total_stages,
            "message": "ML workflow execution started",
        }

        # Process each stage
        for idx, (stage_id, stage_name, duration) in enumerate(self.workflow_stages, 1):
            stage_start = datetime.utcnow()

            # Stage started event
            yield {
                "event": "stage_started",
                "stage_id": stage_id,
                "stage_name": stage_name,
                "stage_number": idx,
                "total_stages": total_stages,
                "progress_percentage": ((idx - 1) / total_stages) * 100,
                "timestamp": stage_start.isoformat(),
            }

            # Simulate stage execution with progress updates
            substeps = 5
            for substep in range(substeps):
                await asyncio.sleep(duration / substeps)

                substep_progress = (substep + 1) / substeps
                overall_progress = ((idx - 1 + substep_progress) / total_stages) * 100

                yield {
                    "event": "stage_progress",
                    "stage_id": stage_id,
                    "stage_name": stage_name,
                    "substep": substep + 1,
                    "total_substeps": substeps,
                    "stage_progress": substep_progress * 100,
                    "overall_progress": overall_progress,
                    "timestamp": datetime.utcnow().isoformat(),
                }

            # Stage completed event
            stage_duration = (datetime.utcnow() - stage_start).total_seconds()

            yield {
                "event": "stage_completed",
                "stage_id": stage_id,
                "stage_name": stage_name,
                "duration_seconds": stage_duration,
                "progress_percentage": (idx / total_stages) * 100,
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Final completion event
        total_duration = (datetime.utcnow() - start_time).total_seconds()

        yield {
            "event": "workflow_completed",
            "status": "success",
            "total_duration_seconds": total_duration,
            "stages_completed": total_stages,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "dataset_processed": True,
                "model_trained": True,
                "accuracy": 0.95,
                "total_time": total_duration,
            },
        }


async def demonstrate_streaming():
    """Demonstrate streaming workflow execution."""
    print("=" * 80)
    print("AG-UI Streaming Workflow Demonstration")
    print("=" * 80)
    print()
    print("This example simulates real-time streaming of ML workflow progress.")
    print("In production, this would connect to the AG-UI FastAPI server.")
    print()
    print("Streaming events:")
    print("  • workflow_started - Initial workflow kickoff")
    print("  • stage_started - Each workflow stage begins")
    print("  • stage_progress - Real-time progress within stage")
    print("  • stage_completed - Stage finishes with metrics")
    print("  • workflow_completed - Final results and summary")
    print()
    print("=" * 80)
    print()

    simulator = StreamingWorkflowSimulator()

    # Stream and display progress
    async for event in simulator.stream_workflow_progress():
        event_type = event["event"]

        if event_type == "workflow_started":
            print(f"🚀 {event['message']}")
            print(f"   Total stages: {event['total_stages']}")
            print()

        elif event_type == "stage_started":
            print(f"\n📍 Stage {event['stage_number']}/{event['total_stages']}: {event['stage_name']}")

        elif event_type == "stage_progress":
            # Show progress bar
            progress = event['stage_progress']
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"   [{bar}] {progress:.1f}% | Overall: {event['overall_progress']:.1f}%", end="\r")

        elif event_type == "stage_completed":
            print()  # New line after progress bar
            print(f"   ✓ Completed in {event['duration_seconds']:.2f}s")

        elif event_type == "workflow_completed":
            print()
            print("=" * 80)
            print("✓ Workflow Completed Successfully")
            print("=" * 80)
            print()
            print(f"Total Duration: {event['total_duration_seconds']:.2f} seconds")
            print(f"Stages Completed: {event['stages_completed']}")
            print()
            print("Summary:")
            summary = event['summary']
            print(f"  • Dataset Processed: {'Yes' if summary['dataset_processed'] else 'No'}")
            print(f"  • Model Trained: {'Yes' if summary['model_trained'] else 'No'}")
            print(f"  • Model Accuracy: {summary['accuracy']:.2%}")
            print(f"  • Total Time: {summary['total_time']:.2f}s")
            print()

    # Save event log
    output_file = project_root / "examples" / "advanced_workflow" / "streaming_events.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Event log would be saved to: {output_file}")
    print()
    print("To use with AG-UI server:")
    print("  1. Start the server: python src/ag_ui_server.py")
    print("  2. Connect to WebSocket: ws://localhost:8000/ws/workflow/{workflow_id}")
    print("  3. Receive real-time events as shown above")
    print()


async def demonstrate_websocket_client():
    """
    Demonstrate WebSocket client connection to AG-UI server.

    This would connect to the actual AG-UI server in production.
    """
    print("=" * 80)
    print("WebSocket Client Example")
    print("=" * 80)
    print()
    print("Example WebSocket client code:")
    print()

    code = '''
import asyncio
import websockets
import json

async def connect_to_workflow_stream(workflow_id: str):
    """Connect to AG-UI workflow stream."""
    uri = f"ws://localhost:8000/ws/workflow/{workflow_id}"

    async with websockets.connect(uri) as websocket:
        print(f"Connected to workflow: {workflow_id}")

        # Receive and process events
        async for message in websocket:
            event = json.loads(message)
            print(f"Event: {event['event']}")

            if event['event'] == 'workflow_completed':
                print("Workflow finished!")
                break

# Run the client
asyncio.run(connect_to_workflow_stream("my-workflow-123"))
    '''

    print(code)
    print()
    print("This client will receive real-time updates during workflow execution.")


def main():
    """Run streaming demonstration."""
    print("Starting Streaming Workflow Demonstration")
    print()

    # Run async demonstration
    asyncio.run(demonstrate_streaming())

    # Show WebSocket client example
    print()
    asyncio.run(demonstrate_websocket_client())

    logger.info("streaming_demo_completed")


if __name__ == "__main__":
    main()
