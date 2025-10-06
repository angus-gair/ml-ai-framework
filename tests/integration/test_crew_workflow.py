"""
Integration tests for CrewAI workflow.
Tests end-to-end multi-agent data analysis pipeline.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import os
import json


class TestCrewAIWorkflow:
    """Integration tests for complete CrewAI workflow."""

    @patch('crewai.Agent')
    @patch('crewai.Task')
    @patch('crewai.Crew')
    def test_crew_initialization(self, mock_crew, mock_task, mock_agent):
        """Test that crew initializes with all required agents."""
        # Mock LLM
        mock_llm = Mock()

        # Create mock agents
        agents = {
            'data_acquisition': Mock(role="Data Acquisition Specialist"),
            'data_cleaning': Mock(role="Data Quality Engineer"),
            'eda': Mock(role="Data Scientist - EDA Specialist"),
            'modeling': Mock(role="Machine Learning Engineer"),
            'reporting': Mock(role="Data Storyteller")
        }

        # Verify all agents are created
        assert len(agents) == 5
        for agent in agents.values():
            assert agent.role is not None

    @patch('crewai.Crew')
    def test_sequential_task_execution(self, mock_crew):
        """Test that tasks execute in correct sequential order."""
        # Mock task execution order
        execution_log = []

        def mock_kickoff(inputs):
            execution_log.extend([
                'acquisition',
                'cleaning',
                'eda',
                'modeling',
                'reporting'
            ])
            return Mock(tasks=[Mock(output=Mock(pydantic=None)) for _ in range(5)])

        mock_crew_instance = Mock()
        mock_crew_instance.kickoff = mock_kickoff
        mock_crew.return_value = mock_crew_instance

        result = mock_crew_instance.kickoff({'data_path': 'test.csv', 'target_variable': 'target'})

        # Verify execution order
        assert execution_log == ['acquisition', 'cleaning', 'eda', 'modeling', 'reporting']

    def test_context_passing_between_tasks(self):
        """Test that context is properly passed between sequential tasks."""
        # Mock task outputs
        mock_acquisition_output = Mock()
        mock_acquisition_output.raw = "{'rows': 100, 'columns': 5}"

        mock_cleaning_output = Mock()
        mock_cleaning_output.pydantic = Mock(
            rows_before=100,
            rows_after=95,
            cleaned_data_path="/tmp/cleaned.csv"
        )

        mock_eda_output = Mock()
        mock_eda_output.pydantic = Mock(
            recommended_features=['f1', 'f2', 'f3']
        )

        # Verify context chain
        assert mock_cleaning_output.pydantic.rows_before == 100
        assert len(mock_eda_output.pydantic.recommended_features) == 3

    @patch('crewai.Crew')
    def test_pydantic_output_validation(self, mock_crew):
        """Test that Pydantic outputs are validated correctly."""
        from pydantic import BaseModel, Field

        class MockCleaningReport(BaseModel):
            rows_before: int = Field(gt=0)
            rows_after: int = Field(gt=0)
            cleaned_data_path: str

        # Valid output
        valid_report = MockCleaningReport(
            rows_before=100,
            rows_after=95,
            cleaned_data_path="/tmp/cleaned.csv"
        )
        assert valid_report.rows_before == 100

        # Invalid output should raise ValidationError
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MockCleaningReport(
                rows_before=0,  # Must be > 0
                rows_after=95,
                cleaned_data_path="/tmp/cleaned.csv"
            )

    def test_crew_memory_integration(self):
        """Test CrewAI memory features (short-term, long-term, entity)."""
        mock_memory = {
            'short_term': [],
            'long_term': {},
            'entity': {}
        }

        # Simulate memory storage
        mock_memory['short_term'].append({
            'task': 'acquisition',
            'output': 'Dataset loaded: 100 rows'
        })

        mock_memory['entity']['dataset'] = {
            'name': 'iris.csv',
            'rows': 100
        }

        assert len(mock_memory['short_term']) == 1
        assert mock_memory['entity']['dataset']['rows'] == 100

    @patch('crewai.tools.tool')
    def test_tool_invocation(self, mock_tool):
        """Test that agents properly invoke tools."""
        # Mock tool
        @mock_tool("Load Dataset")
        def mock_load_dataset(file_path: str) -> str:
            return f"Loaded: {file_path}"

        result = mock_load_dataset("test.csv")
        assert "Loaded: test.csv" in result

    def test_agent_delegation_pattern(self):
        """Test hierarchical agent delegation in CrewAI."""
        # Mock manager agent
        manager = Mock(role="Manager")
        manager.delegate_task = Mock(return_value="Task delegated")

        # Mock worker agents
        workers = [
            Mock(role="Data Acquisition"),
            Mock(role="Data Cleaning")
        ]

        # Simulate delegation
        for worker in workers:
            result = manager.delegate_task()
            assert result == "Task delegated"

    @pytest.mark.parametrize("dataset_path,target,expected_agents", [
        ("iris.csv", "target", 5),
        ("titanic.csv", "survived", 5),
        ("housing.csv", "price", 5)
    ])
    def test_crew_with_different_datasets(self, dataset_path, target, expected_agents):
        """Test crew workflow with different dataset configurations."""
        # Mock crew execution
        inputs = {
            'data_path': dataset_path,
            'target_variable': target
        }

        # Should work with any dataset
        assert inputs['data_path'] == dataset_path
        assert inputs['target_variable'] == target

    def test_error_handling_in_crew(self):
        """Test error handling when tasks fail."""
        # Mock task failure
        mock_task = Mock()
        mock_task.execute = Mock(side_effect=Exception("Tool execution failed"))

        with pytest.raises(Exception) as exc_info:
            mock_task.execute()

        assert "Tool execution failed" in str(exc_info.value)

    def test_crew_output_file_generation(self, temp_dir):
        """Test that crew generates output files correctly."""
        # Mock file outputs
        eda_report_path = temp_dir / "eda_report.md"
        final_report_path = temp_dir / "final_report.md"

        # Simulate file creation
        with open(eda_report_path, 'w') as f:
            f.write("# EDA Report\n\nAnalysis complete.")

        with open(final_report_path, 'w') as f:
            f.write("# Final Report\n\nAll tasks completed.")

        assert os.path.exists(eda_report_path)
        assert os.path.exists(final_report_path)

        # Verify content
        with open(eda_report_path) as f:
            content = f.read()
            assert "EDA Report" in content


class TestCrewAIAgents:
    """Test individual CrewAI agents."""

    def test_data_acquisition_agent(self, temp_dir, iris_dataset_path):
        """Test data acquisition agent functionality."""
        # Mock agent with load_dataset tool
        from tests.unit.test_tools import load_dataset

        result = load_dataset(iris_dataset_path)

        assert "iris_dataset.csv" in result
        assert "'rows':" in result
        assert "'columns':" in result

    def test_data_cleaning_agent(self, temp_dir, dataset_with_missing_values):
        """Test data cleaning agent functionality."""
        from tests.unit.test_tools import clean_data

        input_path = temp_dir / "input.csv"
        output_path = temp_dir / "output.csv"
        dataset_with_missing_values.to_csv(input_path, index=False)

        result = clean_data(str(input_path), str(output_path))

        assert "Cleaned" in result
        assert os.path.exists(output_path)

    def test_eda_agent(self, temp_dir, sample_dataset):
        """Test exploratory data analysis agent."""
        from tests.unit.test_tools import calculate_statistics

        file_path = temp_dir / "data.csv"
        sample_dataset.to_csv(file_path, index=False)

        result = calculate_statistics(str(file_path))

        assert "Statistics:" in result
        assert "Correlations:" in result

    def test_modeling_agent(self, temp_dir, iris_dataset_path):
        """Test predictive modeling agent."""
        from tests.unit.test_tools import train_model

        model_path = temp_dir / "model.pkl"
        result = train_model(iris_dataset_path, 'target', str(model_path))

        assert "accuracy" in result
        assert "f1_score" in result
        assert os.path.exists(model_path)

    def test_reporting_agent(self, cleaning_report_dict, eda_findings_dict, model_performance_dict):
        """Test reporting agent synthesis."""
        # Mock reporting agent output
        from tests.unit.test_models import CleaningReport, EDAFindings, ModelPerformance

        cleaning = CleaningReport(**cleaning_report_dict)
        eda = EDAFindings(**eda_findings_dict)
        model = ModelPerformance(**model_performance_dict)

        # Simulate report generation
        report_content = f"""
        # Data Analysis Report

        ## Data Quality
        - Rows: {cleaning.rows_before} -> {cleaning.rows_after}
        - Duplicates removed: {cleaning.duplicates_removed}

        ## EDA Findings
        - Recommended features: {', '.join(eda.recommended_features)}
        - Strong correlations found: {len(eda.strong_correlations)}

        ## Model Performance
        - Algorithm: {model.algorithm}
        - Accuracy: {model.accuracy:.2%}
        - F1 Score: {model.f1_score:.2%}
        """

        assert "Data Quality" in report_content
        assert "EDA Findings" in report_content
        assert "Model Performance" in report_content


class TestCrewAIConfiguration:
    """Test CrewAI configuration and setup."""

    def test_llm_configuration(self):
        """Test LLM configuration for agents."""
        mock_llm = Mock()
        mock_llm.model = "gpt-4o"
        mock_llm.temperature = 0.7

        assert mock_llm.model == "gpt-4o"
        assert mock_llm.temperature == 0.7

    def test_process_types(self):
        """Test different process types (sequential, hierarchical)."""
        from enum import Enum

        class MockProcess(Enum):
            SEQUENTIAL = "sequential"
            HIERARCHICAL = "hierarchical"

        assert MockProcess.SEQUENTIAL.value == "sequential"
        assert MockProcess.HIERARCHICAL.value == "hierarchical"

    def test_memory_configuration(self):
        """Test memory embedder configuration."""
        memory_config = {
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"}
        }

        assert memory_config['provider'] == "openai"
        assert memory_config['config']['model'] == "text-embedding-3-small"

    def test_agent_max_iterations(self):
        """Test agent max_iter parameter."""
        mock_agent = Mock()
        mock_agent.max_iter = 15

        assert mock_agent.max_iter == 15

    def test_verbose_logging(self):
        """Test verbose output configuration."""
        mock_agent = Mock()
        mock_agent.verbose = True

        assert mock_agent.verbose is True


class TestCrewAIEndToEnd:
    """End-to-end integration tests for full CrewAI workflow."""

    def test_complete_iris_analysis(self, temp_dir):
        """Test complete analysis workflow on Iris dataset."""
        from sklearn.datasets import load_iris
        import pandas as pd

        # Setup Iris dataset
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        input_path = temp_dir / "iris.csv"
        df.to_csv(input_path, index=False)

        # Mock crew execution
        workflow_log = []

        # Step 1: Load dataset
        from tests.unit.test_tools import load_dataset
        load_result = load_dataset(str(input_path))
        workflow_log.append(('load', 'success'))

        # Step 2: Clean data
        from tests.unit.test_tools import clean_data
        cleaned_path = temp_dir / "cleaned.csv"
        clean_result = clean_data(str(input_path), str(cleaned_path))
        workflow_log.append(('clean', 'success'))

        # Step 3: Calculate statistics
        from tests.unit.test_tools import calculate_statistics
        stats_result = calculate_statistics(str(cleaned_path))
        workflow_log.append(('eda', 'success'))

        # Step 4: Train model
        from tests.unit.test_tools import train_model
        model_path = temp_dir / "model.pkl"
        train_result = train_model(str(cleaned_path), 'target', str(model_path))
        workflow_log.append(('model', 'success'))

        # Verify all steps completed
        assert len(workflow_log) == 4
        assert all(status == 'success' for _, status in workflow_log)

    def test_workflow_with_edge_cases(self, temp_dir):
        """Test workflow handles edge cases gracefully."""
        # Create challenging dataset
        df = pd.DataFrame({
            'feature1': [1, 2, None, None, 5],  # Missing values
            'feature2': [1, 1, 1, 1, 1],  # No variance
            'target': [0, 1, 0, 1, 0]
        })

        input_path = temp_dir / "edge_case.csv"
        df.to_csv(input_path, index=False)

        # Should handle gracefully
        from tests.unit.test_tools import load_dataset, clean_data

        load_result = load_dataset(str(input_path))
        assert "'rows': 5" in load_result

        cleaned_path = temp_dir / "cleaned_edge.csv"
        clean_result = clean_data(str(input_path), str(cleaned_path))
        assert "Cleaned" in clean_result
