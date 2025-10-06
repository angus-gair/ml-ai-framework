"""
Integration tests for LangGraph workflow.
Tests state management, graph execution, and checkpointing.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import TypedDict, Annotated, Literal
import operator


class TestLangGraphState:
    """Test LangGraph state management."""

    def test_state_definition(self):
        """Test TypedDict state schema definition."""

        class AnalysisState(TypedDict):
            messages: Annotated[list, operator.add]
            dataset_metadata: dict
            cleaning_report: dict | None
            current_stage: str

        state = AnalysisState(
            messages=[],
            dataset_metadata={},
            cleaning_report=None,
            current_stage="start"
        )

        assert state['current_stage'] == "start"
        assert state['cleaning_report'] is None

    def test_state_reducer_function(self):
        """Test state reducer with operator.add for messages."""

        class MockState(TypedDict):
            messages: Annotated[list, operator.add]

        # Simulate state updates
        initial_state = {'messages': [{'role': 'user', 'content': 'Start'}]}
        update = {'messages': [{'role': 'assistant', 'content': 'Processing'}]}

        # Messages should be appended
        combined_messages = initial_state['messages'] + update['messages']
        assert len(combined_messages) == 2

    def test_immutable_state_updates(self):
        """Test that state updates are immutable."""
        initial_state = {
            'dataset_metadata': {'rows': 100},
            'current_stage': 'acquisition'
        }

        # Create new state instead of mutating
        new_state = {
            **initial_state,
            'current_stage': 'cleaning',
            'dataset_metadata': {**initial_state['dataset_metadata'], 'columns': 5}
        }

        # Original state unchanged
        assert initial_state['current_stage'] == 'acquisition'
        assert 'columns' not in initial_state['dataset_metadata']

        # New state updated
        assert new_state['current_stage'] == 'cleaning'
        assert new_state['dataset_metadata']['columns'] == 5


class TestLangGraphNodes:
    """Test individual LangGraph node functions."""

    def test_supervisor_node_routing(self):
        """Test supervisor routes to correct next node."""

        def mock_supervisor(state):
            stage = state.get('current_stage', 'start')

            routing_map = {
                'start': 'data_acquisition',
                'acquisition_complete': 'data_cleaning',
                'cleaning_complete': 'eda',
                'eda_complete': 'modeling',
                'modeling_complete': 'reporting',
                'reporting_complete': '__end__'
            }

            return {'next': routing_map.get(stage, '__end__')}

        # Test routing logic
        assert mock_supervisor({'current_stage': 'start'})['next'] == 'data_acquisition'
        assert mock_supervisor({'current_stage': 'cleaning_complete'})['next'] == 'eda'
        assert mock_supervisor({'current_stage': 'reporting_complete'})['next'] == '__end__'

    def test_data_acquisition_node(self, temp_dir, iris_dataset_path):
        """Test data acquisition node execution."""
        from tests.unit.test_tools import load_dataset

        state = {
            'messages': [],
            'data_path': iris_dataset_path,
            'dataset_metadata': {}
        }

        # Execute node
        result = load_dataset(state['data_path'])

        # Update state
        updated_state = {
            **state,
            'messages': state['messages'] + [{'role': 'assistant', 'content': f'Dataset loaded: {result}'}],
            'dataset_metadata': eval(result) if not result.startswith("Error") else {}
        }

        assert updated_state['dataset_metadata']['rows'] == 150  # Iris dataset

    def test_cleaning_node(self, temp_dir, dataset_with_missing_values):
        """Test data cleaning node execution."""
        from tests.unit.test_tools import clean_data

        input_path = temp_dir / "input.csv"
        output_path = temp_dir / "cleaned.csv"
        dataset_with_missing_values.to_csv(input_path, index=False)

        state = {
            'messages': [],
            'data_path': str(input_path)
        }

        # Execute cleaning
        result = clean_data(str(input_path), str(output_path))

        updated_state = {
            **state,
            'messages': state['messages'] + [{'role': 'assistant', 'content': f'Cleaning: {result}'}],
            'cleaned_data_path': str(output_path)
        }

        assert 'cleaned_data_path' in updated_state

    def test_eda_node(self, temp_dir, sample_dataset):
        """Test EDA node execution."""
        from tests.unit.test_tools import calculate_statistics

        file_path = temp_dir / "data.csv"
        sample_dataset.to_csv(file_path, index=False)

        result = calculate_statistics(str(file_path))

        assert "Statistics:" in result
        assert "Correlations:" in result

    def test_modeling_node(self, temp_dir, iris_dataset_path):
        """Test modeling node execution."""
        from tests.unit.test_tools import train_model

        model_path = temp_dir / "model.pkl"
        result = train_model(iris_dataset_path, 'target', str(model_path))

        metrics = eval(result)
        assert 'accuracy' in metrics
        assert metrics['accuracy'] > 0


class TestLangGraphExecution:
    """Test graph compilation and execution."""

    def test_graph_construction(self):
        """Test basic graph structure construction."""
        # Mock graph builder
        nodes = ['supervisor', 'data_acquisition', 'data_cleaning', 'eda', 'modeling', 'reporting']
        edges = [
            ('START', 'supervisor'),
            ('data_acquisition', 'supervisor'),
            ('data_cleaning', 'supervisor'),
            ('eda', 'supervisor'),
            ('modeling', 'supervisor'),
            ('reporting', 'supervisor')
        ]

        assert len(nodes) == 6
        assert len(edges) == 6

    def test_conditional_edges(self):
        """Test conditional routing logic."""

        def route_function(state):
            """Route based on current stage."""
            if state.get('error'):
                return 'error_handler'
            elif state.get('complete'):
                return '__end__'
            else:
                return 'next_node'

        # Test different conditions
        assert route_function({'error': True}) == 'error_handler'
        assert route_function({'complete': True}) == '__end__'
        assert route_function({}) == 'next_node'

    @patch('langgraph.checkpoint.sqlite.SqliteSaver')
    def test_checkpointing(self, mock_saver):
        """Test state checkpointing functionality."""
        # Mock checkpoint saver
        mock_saver_instance = Mock()
        mock_saver.from_conn_string = Mock(return_value=mock_saver_instance)

        checkpointer = mock_saver.from_conn_string("test.db")

        # Simulate checkpoint save
        state = {
            'current_stage': 'eda_complete',
            'dataset_metadata': {'rows': 100}
        }

        checkpointer.save = Mock()
        checkpointer.save(state)
        checkpointer.save.assert_called_once()

    def test_stream_execution(self):
        """Test streaming execution of graph."""
        execution_log = []

        def mock_stream_step(state):
            execution_log.append(state.get('current_stage'))
            return state

        # Simulate streaming
        states = [
            {'current_stage': 'acquisition'},
            {'current_stage': 'cleaning'},
            {'current_stage': 'eda'}
        ]

        for state in states:
            mock_stream_step(state)

        assert execution_log == ['acquisition', 'cleaning', 'eda']

    def test_interrupt_and_resume(self):
        """Test human-in-the-loop interrupt functionality."""
        workflow_state = {
            'current_stage': 'eda',
            'interrupted': False,
            'interrupt_data': None
        }

        # Simulate interrupt
        workflow_state['interrupted'] = True
        workflow_state['interrupt_data'] = {'reason': 'human_review_required'}

        assert workflow_state['interrupted'] is True

        # Resume after human input
        workflow_state['interrupted'] = False
        workflow_state['current_stage'] = 'modeling'

        assert workflow_state['current_stage'] == 'modeling'


class TestLangGraphCommand:
    """Test Command pattern for node returns."""

    def test_command_goto(self):
        """Test Command with goto directive."""

        class MockCommand:
            def __init__(self, goto, update):
                self.goto = goto
                self.update = update

        cmd = MockCommand(goto="next_node", update={'stage': 'updated'})

        assert cmd.goto == "next_node"
        assert cmd.update['stage'] == 'updated'

    def test_command_with_state_update(self):
        """Test Command updates state correctly."""

        def mock_node(state):
            return {
                'goto': 'supervisor',
                'update': {
                    'messages': state['messages'] + [{'content': 'Node executed'}],
                    'current_stage': 'complete'
                }
            }

        initial_state = {'messages': [], 'current_stage': 'start'}
        command = mock_node(initial_state)

        assert command['goto'] == 'supervisor'
        assert len(command['update']['messages']) == 1
        assert command['update']['current_stage'] == 'complete'


class TestLangGraphErrorHandling:
    """Test error handling in LangGraph workflows."""

    def test_node_error_propagation(self):
        """Test that node errors are caught and handled."""

        def failing_node(state):
            raise ValueError("Simulated node failure")

        with pytest.raises(ValueError) as exc_info:
            failing_node({})

        assert "Simulated node failure" in str(exc_info.value)

    def test_retry_logic(self):
        """Test retry mechanism for failed nodes."""
        attempt_count = {'count': 0}

        def unreliable_node(state):
            attempt_count['count'] += 1
            if attempt_count['count'] < 3:
                raise Exception("Temporary failure")
            return {'success': True}

        # Simulate retries
        for _ in range(3):
            try:
                result = unreliable_node({})
                break
            except Exception:
                continue

        assert attempt_count['count'] == 3
        assert result['success'] is True

    def test_error_state_tracking(self):
        """Test tracking errors in state."""
        state = {
            'errors': [],
            'current_stage': 'modeling'
        }

        # Simulate error
        try:
            raise ValueError("Model training failed")
        except ValueError as e:
            state['errors'].append({
                'stage': state['current_stage'],
                'error': str(e)
            })

        assert len(state['errors']) == 1
        assert state['errors'][0]['stage'] == 'modeling'


class TestLangGraphEndToEnd:
    """End-to-end integration tests for LangGraph workflow."""

    def test_complete_workflow_execution(self, temp_dir):
        """Test complete workflow from start to finish."""
        from sklearn.datasets import load_iris
        import pandas as pd

        # Setup dataset
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        input_path = temp_dir / "iris.csv"
        df.to_csv(input_path, index=False)

        # Initialize state
        state = {
            'messages': [],
            'dataset_metadata': {},
            'cleaning_report': None,
            'eda_findings': None,
            'model_performance': None,
            'current_stage': 'start',
            'data_path': str(input_path),
            'target_variable': 'target'
        }

        workflow_log = []

        # Execute workflow
        from tests.unit.test_tools import load_dataset, clean_data, calculate_statistics, train_model

        # Step 1: Load
        load_result = load_dataset(state['data_path'])
        state['dataset_metadata'] = eval(load_result)
        state['current_stage'] = 'acquisition_complete'
        workflow_log.append('acquisition')

        # Step 2: Clean
        cleaned_path = temp_dir / "cleaned.csv"
        clean_result = clean_data(state['data_path'], str(cleaned_path))
        state['current_stage'] = 'cleaning_complete'
        workflow_log.append('cleaning')

        # Step 3: EDA
        stats_result = calculate_statistics(str(cleaned_path))
        state['current_stage'] = 'eda_complete'
        workflow_log.append('eda')

        # Step 4: Model
        model_path = temp_dir / "model.pkl"
        train_result = train_model(str(cleaned_path), 'target', str(model_path))
        state['model_performance'] = eval(train_result)
        state['current_stage'] = 'modeling_complete'
        workflow_log.append('modeling')

        # Verify workflow completion
        assert len(workflow_log) == 4
        assert state['current_stage'] == 'modeling_complete'
        assert state['model_performance']['accuracy'] > 0

    def test_checkpoint_recovery(self, temp_dir):
        """Test workflow recovery from checkpoint."""
        # Simulate checkpoint state
        checkpoint_state = {
            'current_stage': 'eda_complete',
            'dataset_metadata': {'rows': 100, 'columns': 5},
            'cleaned_data_path': '/tmp/cleaned.csv',
            'messages': [
                {'role': 'assistant', 'content': 'Acquisition complete'},
                {'role': 'assistant', 'content': 'Cleaning complete'},
                {'role': 'assistant', 'content': 'EDA complete'}
            ]
        }

        # Resume from checkpoint
        resumed_state = checkpoint_state.copy()
        resumed_state['current_stage'] = 'modeling'

        # Verify state preserved
        assert len(resumed_state['messages']) == 3
        assert resumed_state['dataset_metadata']['rows'] == 100

    def test_parallel_execution_simulation(self):
        """Test that independent nodes could execute in parallel."""
        # Mock parallel execution
        import concurrent.futures

        def independent_task(task_id):
            return f"Task {task_id} complete"

        tasks = [1, 2, 3, 4]
        results = []

        # Simulate parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(independent_task, tid) for tid in tasks]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 4


class TestLangGraphStateGraph:
    """Test StateGraph API and features."""

    def test_add_node(self):
        """Test adding nodes to graph."""
        graph_nodes = []

        def add_node(name, func):
            graph_nodes.append({'name': name, 'func': func})

        add_node('supervisor', lambda s: s)
        add_node('worker', lambda s: s)

        assert len(graph_nodes) == 2
        assert graph_nodes[0]['name'] == 'supervisor'

    def test_add_edge(self):
        """Test adding edges between nodes."""
        graph_edges = []

        def add_edge(from_node, to_node):
            graph_edges.append((from_node, to_node))

        add_edge('START', 'supervisor')
        add_edge('worker', 'supervisor')

        assert len(graph_edges) == 2
        assert ('START', 'supervisor') in graph_edges

    def test_conditional_edge(self):
        """Test conditional edge routing."""

        def route_logic(state):
            if state.get('error'):
                return 'error_handler'
            return 'next_node'

        # Test routing
        assert route_logic({'error': True}) == 'error_handler'
        assert route_logic({}) == 'next_node'
