"""Parallel Agent Execution Example.

This example demonstrates:
- Running multiple agents in parallel
- Async workflow execution
- Resource optimization
- Coordinating parallel tasks
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from structlog import get_logger

from src.workflows.crew_system import CrewAIWorkflow
from src.workflows.langgraph_system import LangGraphWorkflow
from src.utils.logging import setup_logging

# Setup logging
setup_logging(log_level="INFO", log_to_file=True, log_dir=str(project_root / "logs"))
logger = get_logger(__name__)


class ParallelWorkflowExecutor:
    """Execute multiple ML workflows in parallel."""

    def __init__(self, max_workers: int = 3):
        """
        Initialize parallel executor.

        Args:
            max_workers: Maximum number of concurrent workflows
        """
        self.max_workers = max_workers
        logger.info("parallel_executor_initialized", workers=max_workers)

    def execute_single_workflow(
        self,
        workflow_type: str,
        data_path: str,
        target_column: str,
        workflow_id: str,
    ) -> Dict[str, Any]:
        """
        Execute a single workflow.

        Args:
            workflow_type: 'crewai' or 'langgraph'
            data_path: Path to dataset
            target_column: Target variable
            workflow_id: Unique workflow identifier

        Returns:
            Workflow results with metadata
        """
        start_time = datetime.utcnow()

        logger.info(
            "workflow_starting",
            workflow_id=workflow_id,
            type=workflow_type,
            data=data_path,
        )

        try:
            if workflow_type == "crewai":
                workflow = CrewAIWorkflow(model_name="gpt-4", temperature=0.7)
                result = workflow.run(data_path, target_column)
            elif workflow_type == "langgraph":
                workflow = LangGraphWorkflow(model_name="gpt-4", temperature=0.7)
                result = workflow.run(data_path, target_column)
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            logger.info(
                "workflow_completed",
                workflow_id=workflow_id,
                duration=execution_time,
                status=result.get("status", "unknown"),
            )

            return {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "status": "success",
                "result": result,
                "execution_time": execution_time,
            }

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            logger.error(
                "workflow_failed",
                workflow_id=workflow_id,
                error=str(e),
                duration=execution_time,
            )

            return {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "status": "failed",
                "error": str(e),
                "execution_time": execution_time,
            }

    def execute_parallel(
        self,
        workflows: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple workflows in parallel.

        Args:
            workflows: List of workflow configurations, each containing:
                - workflow_type: 'crewai' or 'langgraph'
                - data_path: Path to dataset
                - target_column: Target variable
                - workflow_id: Unique identifier

        Returns:
            List of results from all workflows
        """
        start_time = datetime.utcnow()

        logger.info(
            "parallel_execution_starting",
            total_workflows=len(workflows),
            max_workers=self.max_workers,
        )

        print("=" * 80)
        print("Parallel Workflow Execution")
        print("=" * 80)
        print()
        print(f"Workflows to execute: {len(workflows)}")
        print(f"Max parallel workers: {self.max_workers}")
        print()
        print("Workflow Queue:")
        for i, wf in enumerate(workflows, 1):
            print(f"  {i}. [{wf['workflow_id']}] {wf['workflow_type']} - {wf['data_path']}")
        print()
        print("Starting parallel execution...")
        print("-" * 80)

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all workflows
            future_to_workflow = {
                executor.submit(
                    self.execute_single_workflow,
                    wf["workflow_type"],
                    wf["data_path"],
                    wf["target_column"],
                    wf["workflow_id"],
                ): wf
                for wf in workflows
            }

            # Collect results as they complete
            for future in as_completed(future_to_workflow):
                workflow = future_to_workflow[future]
                try:
                    result = future.result()
                    results.append(result)

                    status_symbol = "✓" if result["status"] == "success" else "✗"
                    print(f"{status_symbol} [{result['workflow_id']}] completed in {result['execution_time']:.2f}s")

                except Exception as e:
                    logger.error("workflow_exception", workflow_id=workflow["workflow_id"], error=str(e))
                    results.append({
                        "workflow_id": workflow["workflow_id"],
                        "status": "failed",
                        "error": str(e),
                    })

        total_time = (datetime.utcnow() - start_time).total_seconds()

        print()
        print("-" * 80)
        print(f"Total execution time: {total_time:.2f} seconds")
        print()

        logger.info(
            "parallel_execution_completed",
            total_workflows=len(workflows),
            successful=sum(1 for r in results if r["status"] == "success"),
            failed=sum(1 for r in results if r["status"] == "failed"),
            total_time=total_time,
        )

        return results

    async def execute_async(
        self,
        workflows: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Execute workflows asynchronously (alternative approach).

        Args:
            workflows: List of workflow configurations

        Returns:
            List of results
        """
        logger.info("async_execution_starting", workflows=len(workflows))

        async def run_workflow(wf_config):
            """Async wrapper for workflow execution."""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.execute_single_workflow,
                wf_config["workflow_type"],
                wf_config["data_path"],
                wf_config["target_column"],
                wf_config["workflow_id"],
            )

        # Execute all workflows concurrently
        results = await asyncio.gather(*[run_workflow(wf) for wf in workflows])

        logger.info("async_execution_completed", count=len(results))

        return results


def main():
    """Demonstrate parallel workflow execution."""
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    import pandas as pd

    # Prepare multiple datasets
    print("Preparing datasets for parallel execution...")
    print()

    datasets = []

    # Dataset 1: Iris
    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris['species'] = iris.target
    path_iris = "/tmp/parallel_iris.csv"
    df_iris.to_csv(path_iris, index=False)
    datasets.append(("Iris", path_iris, "species"))

    # Dataset 2: Wine
    wine = load_wine()
    df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
    df_wine['target'] = wine.target
    path_wine = "/tmp/parallel_wine.csv"
    df_wine.to_csv(path_wine, index=False)
    datasets.append(("Wine", path_wine, "target"))

    # Dataset 3: Breast Cancer
    cancer = load_breast_cancer()
    df_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df_cancer['target'] = cancer.target
    path_cancer = "/tmp/parallel_cancer.csv"
    df_cancer.to_csv(path_cancer, index=False)
    datasets.append(("Breast Cancer", path_cancer, "target"))

    for name, path, target in datasets:
        print(f"  ✓ {name}: {path} (target: {target})")

    print()

    # Configure workflows for parallel execution
    workflows = [
        {
            "workflow_id": "iris-crewai",
            "workflow_type": "crewai",
            "data_path": path_iris,
            "target_column": "species",
        },
        {
            "workflow_id": "wine-langgraph",
            "workflow_type": "langgraph",
            "data_path": path_wine,
            "target_column": "target",
        },
        {
            "workflow_id": "cancer-crewai",
            "workflow_type": "crewai",
            "data_path": path_cancer,
            "target_column": "target",
        },
    ]

    # Execute in parallel
    executor = ParallelWorkflowExecutor(max_workers=3)
    results = executor.execute_parallel(workflows)

    # Display summary
    print()
    print("=" * 80)
    print("Parallel Execution Summary")
    print("=" * 80)
    print()

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"Total workflows: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()

    if successful:
        print("Successful Workflows:")
        for r in successful:
            print(f"  ✓ {r['workflow_id']} - {r['execution_time']:.2f}s")
        print()

    if failed:
        print("Failed Workflows:")
        for r in failed:
            print(f"  ✗ {r['workflow_id']} - {r.get('error', 'Unknown error')}")
        print()

    # Calculate speedup
    total_parallel_time = max(r["execution_time"] for r in results if "execution_time" in r)
    total_sequential_time = sum(r["execution_time"] for r in results if "execution_time" in r)

    print("Performance Analysis:")
    print(f"  Sequential time (estimated): {total_sequential_time:.2f}s")
    print(f"  Parallel time: {total_parallel_time:.2f}s")
    print(f"  Speedup: {total_sequential_time / total_parallel_time:.2f}x")
    print()

    # Save results
    output_file = project_root / "examples" / "advanced_workflow" / "parallel_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("Parallel Workflow Execution Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Workflows: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Speedup: {total_sequential_time / total_parallel_time:.2f}x\n\n")

        for r in results:
            f.write(f"\n{r['workflow_id']}:\n")
            f.write(f"  Status: {r['status']}\n")
            f.write(f"  Time: {r.get('execution_time', 'N/A')}s\n")
            if r["status"] == "success":
                f.write(f"  Result: {r.get('result', {})}\n")

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
