"""
Unit tests for Pydantic models with comprehensive validation edge cases.
Tests ensure strict type safety and data validation at agent boundaries.
"""
import pytest
from pydantic import ValidationError
from datetime import datetime
import uuid


# Import models - these will be created by the coder agent
# For testing purposes, we'll define them here with the exact schema from spec
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Optional, Literal


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
    missing_values_handled: Dict[str, str]
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
    visualizations_created: List[str]


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


class TestDatasetMetadata:
    """Test suite for DatasetMetadata model validation."""

    def test_valid_dataset_metadata(self, dataset_metadata_dict):
        """Test creation with valid data."""
        metadata = DatasetMetadata(**dataset_metadata_dict)
        assert metadata.name == "test_dataset.csv"
        assert metadata.rows == 100
        assert metadata.columns == 5
        assert len(metadata.dtypes) == 5

    def test_auto_generated_dataset_id(self, dataset_metadata_dict):
        """Test automatic UUID generation for dataset_id."""
        data = dataset_metadata_dict.copy()
        del data['dataset_id']
        metadata = DatasetMetadata(**data)
        assert metadata.dataset_id is not None
        assert len(metadata.dataset_id) == 36  # UUID format

    def test_auto_generated_timestamp(self, dataset_metadata_dict):
        """Test automatic timestamp generation."""
        metadata = DatasetMetadata(**dataset_metadata_dict)
        assert isinstance(metadata.timestamp, datetime)
        assert metadata.timestamp <= datetime.now()

    def test_empty_name_rejected(self, dataset_metadata_dict):
        """Test that empty name is rejected."""
        data = dataset_metadata_dict.copy()
        data['name'] = ""
        with pytest.raises(ValidationError) as exc_info:
            DatasetMetadata(**data)
        assert 'name' in str(exc_info.value)

    def test_zero_rows_rejected(self, dataset_metadata_dict):
        """Test that zero or negative rows are rejected."""
        data = dataset_metadata_dict.copy()
        data['rows'] = 0
        with pytest.raises(ValidationError) as exc_info:
            DatasetMetadata(**data)
        assert 'rows' in str(exc_info.value)

    def test_negative_rows_rejected(self, dataset_metadata_dict):
        """Test that negative rows are rejected."""
        data = dataset_metadata_dict.copy()
        data['rows'] = -5
        with pytest.raises(ValidationError):
            DatasetMetadata(**data)

    def test_zero_columns_rejected(self, dataset_metadata_dict):
        """Test that zero columns are rejected."""
        data = dataset_metadata_dict.copy()
        data['columns'] = 0
        with pytest.raises(ValidationError):
            DatasetMetadata(**data)

    def test_missing_values_default(self, dataset_metadata_dict):
        """Test that missing_values defaults to empty dict."""
        data = dataset_metadata_dict.copy()
        del data['missing_values']
        metadata = DatasetMetadata(**data)
        assert metadata.missing_values == {}

    def test_strict_mode_rejects_extra_fields(self, dataset_metadata_dict):
        """Test that strict mode rejects unknown fields."""
        data = dataset_metadata_dict.copy()
        data['extra_field'] = 'should_fail'
        with pytest.raises(ValidationError):
            DatasetMetadata(**data)


class TestCleaningReport:
    """Test suite for CleaningReport model validation."""

    def test_valid_cleaning_report(self, cleaning_report_dict):
        """Test creation with valid data."""
        report = CleaningReport(**cleaning_report_dict)
        assert report.rows_before == 100
        assert report.rows_after == 95
        assert report.duplicates_removed == 5

    def test_rows_after_exceeds_before_rejected(self, cleaning_report_dict):
        """Test validation that rows_after cannot exceed rows_before."""
        data = cleaning_report_dict.copy()
        data['rows_after'] = 105  # More than rows_before (100)
        with pytest.raises(ValidationError) as exc_info:
            CleaningReport(**data)
        assert 'rows_after cannot exceed rows_before' in str(exc_info.value)

    def test_rows_after_equals_before_accepted(self, cleaning_report_dict):
        """Test that rows_after can equal rows_before (no rows removed)."""
        data = cleaning_report_dict.copy()
        data['rows_after'] = 100
        data['duplicates_removed'] = 0
        report = CleaningReport(**data)
        assert report.rows_after == report.rows_before

    def test_negative_duplicates_rejected(self, cleaning_report_dict):
        """Test that negative duplicates_removed is rejected."""
        data = cleaning_report_dict.copy()
        data['duplicates_removed'] = -5
        with pytest.raises(ValidationError):
            CleaningReport(**data)

    def test_zero_rows_before_rejected(self, cleaning_report_dict):
        """Test that zero rows_before is rejected."""
        data = cleaning_report_dict.copy()
        data['rows_before'] = 0
        with pytest.raises(ValidationError):
            CleaningReport(**data)

    def test_feature_engineering_default_empty_list(self, cleaning_report_dict):
        """Test that feature_engineering defaults to empty list."""
        data = cleaning_report_dict.copy()
        del data['feature_engineering']
        report = CleaningReport(**data)
        assert report.feature_engineering == []

    def test_empty_missing_values_handled_accepted(self, cleaning_report_dict):
        """Test that empty missing_values_handled dict is valid."""
        data = cleaning_report_dict.copy()
        data['missing_values_handled'] = {}
        report = CleaningReport(**data)
        assert report.missing_values_handled == {}


class TestCorrelationPair:
    """Test suite for CorrelationPair model validation."""

    def test_valid_correlation_pair(self):
        """Test creation with valid correlation."""
        pair = CorrelationPair(
            feature_a="feature1",
            feature_b="feature2",
            correlation=0.85,
            significance="strong"
        )
        assert pair.correlation == 0.85
        assert pair.significance == "strong"

    def test_correlation_boundary_values(self):
        """Test correlation at boundary values -1 and 1."""
        pair1 = CorrelationPair(
            feature_a="a", feature_b="b",
            correlation=-1.0, significance="strong"
        )
        assert pair1.correlation == -1.0

        pair2 = CorrelationPair(
            feature_a="a", feature_b="b",
            correlation=1.0, significance="strong"
        )
        assert pair2.correlation == 1.0

    def test_correlation_out_of_bounds_rejected(self):
        """Test that correlation outside [-1, 1] is rejected."""
        with pytest.raises(ValidationError):
            CorrelationPair(
                feature_a="a", feature_b="b",
                correlation=1.5, significance="strong"
            )

        with pytest.raises(ValidationError):
            CorrelationPair(
                feature_a="a", feature_b="b",
                correlation=-1.5, significance="weak"
            )

    def test_invalid_significance_rejected(self):
        """Test that invalid significance literal is rejected."""
        with pytest.raises(ValidationError):
            CorrelationPair(
                feature_a="a", feature_b="b",
                correlation=0.5, significance="invalid"
            )

    @pytest.mark.parametrize("significance", ["weak", "moderate", "strong"])
    def test_all_significance_levels_accepted(self, significance):
        """Test all valid significance levels."""
        pair = CorrelationPair(
            feature_a="a", feature_b="b",
            correlation=0.5, significance=significance
        )
        assert pair.significance == significance


class TestEDAFindings:
    """Test suite for EDAFindings model validation."""

    def test_valid_eda_findings(self, eda_findings_dict):
        """Test creation with valid EDA data."""
        findings = EDAFindings(**eda_findings_dict)
        assert len(findings.recommended_features) == 3
        assert len(findings.strong_correlations) == 1

    def test_empty_recommended_features_rejected(self, eda_findings_dict):
        """Test that empty recommended_features is rejected."""
        data = eda_findings_dict.copy()
        data['recommended_features'] = []
        with pytest.raises(ValidationError) as exc_info:
            EDAFindings(**data)
        assert 'recommended_features' in str(exc_info.value)

    def test_empty_strong_correlations_accepted(self, eda_findings_dict):
        """Test that empty strong_correlations list is valid."""
        data = eda_findings_dict.copy()
        data['strong_correlations'] = []
        findings = EDAFindings(**data)
        assert findings.strong_correlations == []

    def test_correlation_pair_validation(self, eda_findings_dict):
        """Test that CorrelationPair items are validated."""
        data = eda_findings_dict.copy()
        data['strong_correlations'] = [
            {
                "feature_a": "f1",
                "feature_b": "f2",
                "correlation": 2.0,  # Invalid
                "significance": "strong"
            }
        ]
        with pytest.raises(ValidationError):
            EDAFindings(**data)

    def test_nested_statistics_structure(self, eda_findings_dict):
        """Test nested dictionary validation for summary_statistics."""
        findings = EDAFindings(**eda_findings_dict)
        assert 'feature1' in findings.summary_statistics
        assert 'mean' in findings.summary_statistics['feature1']


class TestModelPerformance:
    """Test suite for ModelPerformance model validation."""

    def test_valid_model_performance(self, model_performance_dict):
        """Test creation with valid metrics."""
        perf = ModelPerformance(**model_performance_dict)
        assert perf.algorithm == "RandomForestClassifier"
        assert perf.accuracy == 0.92
        assert perf.f1_score == 0.89

    @pytest.mark.parametrize("metric,value", [
        ("accuracy", 1.5),
        ("precision", -0.1),
        ("recall", 2.0),
        ("f1_score", -1.0)
    ])
    def test_metrics_out_of_bounds_rejected(self, model_performance_dict, metric, value):
        """Test that metrics outside [0, 1] are rejected."""
        data = model_performance_dict.copy()
        data[metric] = value
        with pytest.raises(ValidationError):
            ModelPerformance(**data)

    @pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "f1_score"])
    def test_metrics_boundary_values(self, model_performance_dict, metric):
        """Test metrics at boundary values 0 and 1."""
        data = model_performance_dict.copy()

        # Test lower bound
        data[metric] = 0.0
        perf = ModelPerformance(**data)
        assert getattr(perf, metric) == 0.0

        # Test upper bound
        data[metric] = 1.0
        perf = ModelPerformance(**data)
        assert getattr(perf, metric) == 1.0

    def test_confusion_matrix_optional(self, model_performance_dict):
        """Test that confusion_matrix is optional."""
        data = model_performance_dict.copy()
        del data['confusion_matrix']
        perf = ModelPerformance(**data)
        assert perf.confusion_matrix is None

    def test_confusion_matrix_structure(self, model_performance_dict):
        """Test confusion matrix structure validation."""
        data = model_performance_dict.copy()
        data['confusion_matrix'] = [[10, 5], [3, 12]]
        perf = ModelPerformance(**data)
        assert len(perf.confusion_matrix) == 2
        assert len(perf.confusion_matrix[0]) == 2

    def test_feature_importance_sums_validation(self, model_performance_dict):
        """Test feature importance dictionary validation."""
        perf = ModelPerformance(**model_performance_dict)
        assert isinstance(perf.feature_importance, dict)
        assert sum(perf.feature_importance.values()) <= 1.01  # Allow small float error


class TestAnalysisReport:
    """Test suite for AnalysisReport composite model."""

    def test_valid_analysis_report(
        self,
        cleaning_report_dict,
        eda_findings_dict,
        model_performance_dict
    ):
        """Test creation of complete analysis report."""
        report = AnalysisReport(
            dataset_name="iris_dataset",
            data_quality=CleaningReport(**cleaning_report_dict),
            eda_findings=EDAFindings(**eda_findings_dict),
            model_performance=ModelPerformance(**model_performance_dict),
            executive_summary="This is a comprehensive analysis report " * 10,
            key_insights=[
                "Insight 1: Strong correlation found",
                "Insight 2: Model performs well",
                "Insight 3: Data quality is good"
            ],
            recommendations=[
                "Recommendation 1: Deploy model",
                "Recommendation 2: Monitor performance",
                "Recommendation 3: Collect more data"
            ]
        )
        assert report.dataset_name == "iris_dataset"
        assert len(report.key_insights) == 3
        assert len(report.recommendations) == 3

    def test_executive_summary_min_length(
        self,
        cleaning_report_dict,
        eda_findings_dict,
        model_performance_dict
    ):
        """Test that executive_summary enforces minimum length."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisReport(
                dataset_name="test",
                data_quality=CleaningReport(**cleaning_report_dict),
                eda_findings=EDAFindings(**eda_findings_dict),
                model_performance=ModelPerformance(**model_performance_dict),
                executive_summary="Too short",  # Less than 100 chars
                key_insights=["a", "b", "c"],
                recommendations=["x", "y", "z"]
            )
        assert 'executive_summary' in str(exc_info.value)

    def test_insufficient_insights_rejected(
        self,
        cleaning_report_dict,
        eda_findings_dict,
        model_performance_dict
    ):
        """Test that less than 3 insights is rejected."""
        with pytest.raises(ValidationError):
            AnalysisReport(
                dataset_name="test",
                data_quality=CleaningReport(**cleaning_report_dict),
                eda_findings=EDAFindings(**eda_findings_dict),
                model_performance=ModelPerformance(**model_performance_dict),
                executive_summary="A" * 100,
                key_insights=["insight1", "insight2"],  # Only 2
                recommendations=["r1", "r2", "r3"]
            )

    def test_insufficient_recommendations_rejected(
        self,
        cleaning_report_dict,
        eda_findings_dict,
        model_performance_dict
    ):
        """Test that less than 3 recommendations is rejected."""
        with pytest.raises(ValidationError):
            AnalysisReport(
                dataset_name="test",
                data_quality=CleaningReport(**cleaning_report_dict),
                eda_findings=EDAFindings(**eda_findings_dict),
                model_performance=ModelPerformance(**model_performance_dict),
                executive_summary="A" * 100,
                key_insights=["i1", "i2", "i3"],
                recommendations=["r1"]  # Only 1
            )

    def test_nested_model_validation(
        self,
        cleaning_report_dict,
        eda_findings_dict,
        model_performance_dict
    ):
        """Test that nested model validation works correctly."""
        # Invalid nested CleaningReport
        invalid_cleaning = cleaning_report_dict.copy()
        invalid_cleaning['rows_after'] = 200  # Exceeds rows_before

        with pytest.raises(ValidationError):
            AnalysisReport(
                dataset_name="test",
                data_quality=CleaningReport(**invalid_cleaning),
                eda_findings=EDAFindings(**eda_findings_dict),
                model_performance=ModelPerformance(**model_performance_dict),
                executive_summary="A" * 100,
                key_insights=["i1", "i2", "i3"],
                recommendations=["r1", "r2", "r3"]
            )
