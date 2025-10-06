"""
Mocked LLM tests for deterministic agent testing.
Uses unittest.mock and Pydantic AI TestModel for reproducible tests.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
import json


class TestMockedLLMCalls:
    """Test agents with mocked LLM responses."""

    @patch('openai.OpenAI')
    def test_mock_openai_completion(self, mock_openai):
        """Test mocking OpenAI completion calls."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Mocked analysis result"
        mock_response.usage = Mock(total_tokens=100, prompt_tokens=50, completion_tokens=50)

        mock_client.chat.completions.create = Mock(return_value=mock_response)
        mock_openai.return_value = mock_client

        # Execute
        client = mock_openai()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Analyze this data"}]
        )

        # Verify
        assert response.choices[0].message.content == "Mocked analysis result"
        assert response.usage.total_tokens == 100
        mock_client.chat.completions.create.assert_called_once()

    @patch('anthropic.Anthropic')
    def test_mock_anthropic_completion(self, mock_anthropic):
        """Test mocking Anthropic Claude completion calls."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Claude's mocked response"
        mock_response.usage = Mock(input_tokens=50, output_tokens=50)

        mock_client.messages.create = Mock(return_value=mock_response)
        mock_anthropic.return_value = mock_client

        # Execute
        client = mock_anthropic()
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Analyze this"}]
        )

        # Verify
        assert response.content[0].text == "Claude's mocked response"
        assert response.usage.input_tokens == 50

    def test_mock_llm_with_side_effects(self):
        """Test LLM mock with different responses for sequential calls."""
        mock_llm = Mock()

        # Configure side effects for multiple calls
        mock_llm.generate.side_effect = [
            "First response",
            "Second response",
            "Third response"
        ]

        # Execute multiple calls
        assert mock_llm.generate() == "First response"
        assert mock_llm.generate() == "Second response"
        assert mock_llm.generate() == "Third response"

        # Verify call count
        assert mock_llm.generate.call_count == 3

    def test_mock_llm_with_structured_output(self):
        """Test LLM mock returning structured Pydantic output."""
        from pydantic import BaseModel

        class AnalysisOutput(BaseModel):
            summary: str
            confidence: float

        mock_llm = Mock()
        mock_llm.with_structured_output = Mock(return_value=Mock())
        mock_llm.with_structured_output().invoke = Mock(
            return_value=AnalysisOutput(summary="Data looks good", confidence=0.95)
        )

        # Execute
        structured_llm = mock_llm.with_structured_output()
        result = structured_llm.invoke("Analyze data")

        # Verify
        assert isinstance(result, AnalysisOutput)
        assert result.summary == "Data looks good"
        assert result.confidence == 0.95


class TestPydanticAITestModel:
    """Test using Pydantic AI TestModel for deterministic testing."""

    def test_test_model_basic(self):
        """Test basic TestModel usage."""
        # Simulate Pydantic AI TestModel
        class TestModel:
            def __init__(self, responses):
                self.responses = responses
                self.call_count = 0

            def __call__(self, prompt):
                response = self.responses[self.call_count % len(self.responses)]
                self.call_count += 1
                return response

        test_model = TestModel([
            "Dataset loaded successfully",
            "Data cleaning complete",
            "EDA findings generated"
        ])

        assert test_model("Load data") == "Dataset loaded successfully"
        assert test_model("Clean data") == "Data cleaning complete"
        assert test_model("Analyze data") == "EDA findings generated"

    def test_test_model_with_validation(self):
        """Test TestModel with Pydantic validation."""
        from pydantic import BaseModel, ValidationError

        class DataReport(BaseModel):
            rows: int
            columns: int
            quality_score: float

        # Mock model that returns valid Pydantic data
        def mock_model_output():
            return DataReport(rows=100, columns=5, quality_score=0.95)

        result = mock_model_output()
        assert result.rows == 100
        assert result.quality_score == 0.95

    def test_test_model_deterministic_output(self):
        """Test that TestModel provides deterministic outputs."""
        # Same input should always produce same output
        mock_responses = {
            "analyze dataset": "Dataset contains 100 rows with 5 features",
            "calculate stats": "Mean: 50.5, Std: 28.87"
        }

        def deterministic_llm(prompt):
            return mock_responses.get(prompt, "Unknown query")

        # Multiple calls with same input
        assert deterministic_llm("analyze dataset") == "Dataset contains 100 rows with 5 features"
        assert deterministic_llm("analyze dataset") == "Dataset contains 100 rows with 5 features"
        assert deterministic_llm("calculate stats") == "Mean: 50.5, Std: 28.87"


class TestAgentWithMockedLLM:
    """Test agent behavior with mocked LLM responses."""

    def test_data_acquisition_agent_mock(self, mock_llm):
        """Test data acquisition agent with mocked LLM."""
        # Configure mock LLM response
        mock_llm.invoke.return_value.content = """
        I will load the dataset using the load_dataset tool.
        The dataset contains 150 rows and 5 columns.
        """

        # Simulate agent execution
        agent_output = mock_llm.invoke("Load the dataset from path")

        assert "150 rows" in agent_output.content
        assert "5 columns" in agent_output.content

    def test_cleaning_agent_mock(self, mock_llm):
        """Test cleaning agent with mocked LLM."""
        mock_llm.invoke.return_value.content = """
        I have cleaned the dataset:
        - Removed 5 duplicate rows
        - Imputed 10 missing values using median
        - Detected 3 outliers in feature1
        """

        agent_output = mock_llm.invoke("Clean the dataset")

        assert "duplicate" in agent_output.content.lower()
        assert "missing values" in agent_output.content.lower()

    def test_eda_agent_mock(self, mock_llm):
        """Test EDA agent with mocked LLM."""
        mock_llm.invoke.return_value.content = """
        Exploratory Data Analysis Results:
        - Strong correlation (0.95) between feature1 and feature2
        - feature3 shows normal distribution
        - Recommended features: feature1, feature2, feature3
        """

        agent_output = mock_llm.invoke("Perform EDA")

        assert "correlation" in agent_output.content.lower()
        assert "recommended features" in agent_output.content.lower()

    def test_modeling_agent_mock(self, mock_llm):
        """Test modeling agent with mocked LLM."""
        mock_llm.invoke.return_value.content = """
        Model Training Complete:
        - Algorithm: RandomForestClassifier
        - Accuracy: 95.3%
        - F1-Score: 94.8%
        - Top features: feature1 (35%), feature2 (30%), feature3 (25%)
        """

        agent_output = mock_llm.invoke("Train the model")

        assert "95.3%" in agent_output.content
        assert "RandomForestClassifier" in agent_output.content


class TestToolCallMocking:
    """Test mocking of tool calls within agent execution."""

    def test_mock_tool_execution(self):
        """Test mocking individual tool calls."""
        # Mock tool
        mock_load_tool = Mock(return_value="{'rows': 100, 'columns': 5}")

        result = mock_load_tool("data.csv")

        assert "'rows': 100" in result
        mock_load_tool.assert_called_once_with("data.csv")

    def test_mock_multiple_tool_calls(self):
        """Test mocking sequence of tool calls."""
        mock_tools = {
            'load_dataset': Mock(return_value="Dataset loaded"),
            'clean_data': Mock(return_value="Data cleaned"),
            'calculate_statistics': Mock(return_value="Stats calculated")
        }

        # Simulate workflow
        results = []
        results.append(mock_tools['load_dataset']("data.csv"))
        results.append(mock_tools['clean_data']("data.csv", "cleaned.csv"))
        results.append(mock_tools['calculate_statistics']("cleaned.csv"))

        assert len(results) == 3
        assert results[0] == "Dataset loaded"
        assert results[2] == "Stats calculated"

    def test_mock_tool_with_exceptions(self):
        """Test tool mock that raises exceptions."""
        mock_tool = Mock(side_effect=FileNotFoundError("File not found"))

        with pytest.raises(FileNotFoundError):
            mock_tool("nonexistent.csv")

    def test_mock_tool_call_verification(self):
        """Test verifying tool was called with correct arguments."""
        mock_train = Mock()

        mock_train("data.csv", "target", "model.pkl")

        mock_train.assert_called_once_with("data.csv", "target", "model.pkl")


class TestLLMResponsePatterns:
    """Test different LLM response patterns."""

    def test_mock_streaming_response(self):
        """Test mocking streaming LLM responses."""

        def mock_stream():
            chunks = [
                "Analyzing",
                " the",
                " dataset",
                "...",
                " Complete!"
            ]
            for chunk in chunks:
                yield chunk

        # Collect streamed response
        full_response = "".join(mock_stream())

        assert full_response == "Analyzing the dataset... Complete!"

    def test_mock_function_calling(self):
        """Test mocking LLM function/tool calling."""
        mock_llm = Mock()

        # Mock function call response
        mock_llm.invoke.return_value = Mock(
            tool_calls=[
                {
                    'name': 'load_dataset',
                    'arguments': {'file_path': 'data.csv'}
                }
            ]
        )

        response = mock_llm.invoke("Load the dataset")

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]['name'] == 'load_dataset'

    def test_mock_retry_logic(self):
        """Test mocking retry logic for failed LLM calls."""
        mock_llm = Mock()

        # First two calls fail, third succeeds
        mock_llm.invoke.side_effect = [
            Exception("Rate limit"),
            Exception("Rate limit"),
            Mock(content="Success")
        ]

        # Simulate retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = mock_llm.invoke("prompt")
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise

        assert result.content == "Success"
        assert mock_llm.invoke.call_count == 3


class TestMockLLMCostTracking:
    """Test tracking LLM costs with mocked calls."""

    def test_token_usage_tracking(self):
        """Test tracking token usage across mock calls."""
        token_usage = {'total': 0, 'prompt': 0, 'completion': 0}

        def mock_llm_call(prompt, tokens=100):
            token_usage['total'] += tokens
            token_usage['prompt'] += tokens // 2
            token_usage['completion'] += tokens // 2
            return f"Response using {tokens} tokens"

        # Simulate multiple calls
        mock_llm_call("First prompt", 100)
        mock_llm_call("Second prompt", 150)
        mock_llm_call("Third prompt", 200)

        assert token_usage['total'] == 450
        assert token_usage['prompt'] == 225
        assert token_usage['completion'] == 225

    def test_cost_estimation(self):
        """Test estimating costs from token usage."""
        # GPT-4o pricing (example rates)
        PRICE_PER_1K_PROMPT = 0.005
        PRICE_PER_1K_COMPLETION = 0.015

        def estimate_cost(prompt_tokens, completion_tokens):
            prompt_cost = (prompt_tokens / 1000) * PRICE_PER_1K_PROMPT
            completion_cost = (completion_tokens / 1000) * PRICE_PER_1K_COMPLETION
            return prompt_cost + completion_cost

        cost = estimate_cost(prompt_tokens=5000, completion_tokens=2000)

        assert cost == pytest.approx(0.055, 0.001)


class TestMockLLMIntegration:
    """Integration tests with mocked LLM across entire workflow."""

    def test_end_to_end_workflow_with_mock(self, mock_llm):
        """Test complete workflow with all LLM calls mocked."""
        # Configure mock responses for each stage
        responses = [
            "Dataset loaded: 150 rows, 5 columns",
            "Data cleaned: removed 0 duplicates, handled 0 missing values",
            "EDA complete: identified strong correlations",
            "Model trained: 95% accuracy achieved",
            "Report generated: comprehensive analysis complete"
        ]

        mock_llm.invoke.side_effect = [Mock(content=r) for r in responses]

        # Execute workflow
        workflow_log = []

        stages = ['acquisition', 'cleaning', 'eda', 'modeling', 'reporting']
        for stage in stages:
            result = mock_llm.invoke(f"{stage} prompt")
            workflow_log.append(result.content)

        # Verify all stages executed
        assert len(workflow_log) == 5
        assert "Dataset loaded" in workflow_log[0]
        assert "Model trained" in workflow_log[3]
        assert "Report generated" in workflow_log[4]

    def test_mock_agent_collaboration(self, mock_llm):
        """Test mocked collaboration between multiple agents."""
        # Mock different agents
        acquisition_agent = Mock()
        acquisition_agent.execute = Mock(return_value="Data acquired")

        cleaning_agent = Mock()
        cleaning_agent.execute = Mock(return_value="Data cleaned")

        # Simulate collaboration
        result1 = acquisition_agent.execute()
        result2 = cleaning_agent.execute()

        assert result1 == "Data acquired"
        assert result2 == "Data cleaned"

        # Verify both agents called
        acquisition_agent.execute.assert_called_once()
        cleaning_agent.execute.assert_called_once()
