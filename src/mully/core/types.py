from uuid import uuid4
from io import BytesIO
from datetime import datetime
from typing import Union, Iterable, Self
from pydantic import (
    BaseModel,
    Field,
    computed_field,
    model_validator
)
from mully.core.enums import (
    DatasetType,
    DatasetCountry,
    DatasetSector,
    DatasetLanguage,
    TaskType,
    MetricType,
    GenerationProvider,
    ClassificationProvider,
    ClusteringProvider,
    RegressionProvider,
    TabularOutputFormat,
    DatabaseOutputFormat,
    DocumentOutputFormat,
    AudioOutputFormat,
    ImageOutputFormat,
    VideoOutputFormat,
)


PROVIDER = Union[
    GenerationProvider,
    ClassificationProvider,
    ClusteringProvider,
    RegressionProvider
]
OUTPUT_FORMAT = Union[
    TabularOutputFormat,
    DatabaseOutputFormat,
    DocumentOutputFormat,
    AudioOutputFormat,
    ImageOutputFormat,
    VideoOutputFormat,
]

def new_id():
    """_summary_"""
    return str(uuid4()).replace("-", "")


class MullyBase(BaseModel):
    """_summary_: Class for Mully base"""

    # TODO: Implement Datetime Serializer

    class Config:
        """_summary_: Pydantic Config class for Hyplate Models."""

        arbitrary_types_allowed = True
        use_enum_values = True
        validate_assignment = True
        evaluation_error_cause = True

    id: str = Field(
        default_factory=new_id,
        title="Object ID",
        description="Unique Identifier",
        example="123e4567-e89b-12d3-a456-426614174000",
    )
    created_at: str = Field(  # TODO: Implement Datetime Serializer
        default=datetime.now().isoformat(),
        title="Created At",
        description="Creation Timestamp",
        example="2021-09-01T00:00:00Z",
    )


def get_output_model(properties: list[tuple[str, type]]) -> type[MullyBase]:
    """_summary_: Dynamically Create Pydantic Models for Output"""

    class NewModel(MullyBase):
        """_summary_: Dynamically Create Pydantic Models for Output"""
        __annotations__ = {prop[0]: prop[1] for prop in properties}
    return NewModel


class Source(MullyBase):
    """_summary_: _description_"""
    name: str = Field(
        ...,
        title="Name",
        description="Name of the source",
        example="source_name",
    )
    url: str = Field(
        ...,
        title="URL",
        description="URL of the source",
        example="https://www.example.com",
    )
    #todo: Add Source article Authors

class Violation(MullyBase):
    """_summary_: Base Model for violations"""

    dataset_id: str = Field(
        ...,
        title="Dataset ID",
        description="ID of the dataset",
        example="123e4567-e89b-12d3-a456-426614174000",
    )
    description: str = Field(
        ..., title="Description", description="Description of the violation"
    )
    severity: int = Field(
        ..., title="Severity", description="Severity of the violation"
    )
    category: str = Field(
        ..., title="Category", description="Category of the violation"
    )
    source: str = Field(..., title="Source", description="Source of the violation")
    rule: str = Field(..., title="Rule", description="Rule of the violation")
    location: str = Field(
        ..., title="Location", description="Location of the violation"
    )
    solution: str = Field(
        ..., title="Solution", description="Solution of the violation"
    )


class FeatureRequest(MullyBase):
    """_summary_: Base Model for Creating Feature Requests"""
    dataset_task: TaskType = Field(
        ...,
        title="Dataset Task",
        description="Task of the dataset",
        example=TaskType.get_values(),
    )
    dataset_type: DatasetType = Field(
        ...,
        title="Dataset Type",
        description="Type of the dataset",
        example=DatasetType.get_values(),
    )
    dataset_goal: str = Field(
        ...,
        title="Dataset Goal",
        description="Goal of the dataset",
        example="To predict ......., To build ......., To test .......",
    )
    dataset_description: str = Field(
        ...,
        title="Dataset Description",
        description="Description of the dataset",
        example="A Dataset of .......",
    )
    dataset_country: DatasetCountry = Field(
        default=DatasetCountry.ANY.value,
        title="Dataset DatasetCountry",
        description="DatasetCountry of the dataset",
        example=DatasetCountry.get_values(),
    )
    dataset_sector: DatasetSector = Field(
        default=DatasetSector.ANY.value,
        title="Dataset Business Domain",
        description="Business domain of the dataset",
        example=DatasetSector.get_values(),
    )
    dataset_language: DatasetLanguage = Field(
        default=DatasetLanguage.ENGLISH.value,
        title="Dataset DatasetLanguage",
        description="DatasetLanguage of the dataset",
        example=DatasetLanguage.get_values(),
    )

    @computed_field
    @property
    def log_dir(self) -> str:
        """_summary_: Log Directory"""
        return f"./.logs/{self.id}/"

class BaseFeature(MullyBase):
    """_summary_: _description_"""
    dataset_id: str = Field(
        ...,
        title="Dataset ID",
        description="ID of the dataset",
        example="123e4567-e89b-12d3-a456-426614174000",
    )
    feature_name: str = Field(
        ...,
        title="Feature Name",
        description="Name of the feature",
        example="transaction_amount"
    )
    feature_description: str = Field(
        ...,
        title="Feature Description",
        description="Description of the feature",
        example="Amount of the transaction"
    )
    feature_type: str = Field(
        ...,
        title="Feature Type",
        description="Type of the feature",
        example="numeric"
    )
    feature_sub_type: str = Field(
        ...,
        title="Feature Sub Type",
        description="Sub type of the feature",
        example="decimal"
    )
    feature_rank: int = Field(
        ...,
        title="Rank",
        description="Importance Rank of the feature relative to parent dataset",
        example=1
    )
    is_target: bool = Field(
        ...,
        title="Is Target",
        description="Is the feature the target"
    )


class GuidelineRequest(FeatureRequest):
    """_summary_: Request Class For Creating Guidelines"""
    dataset_features: Iterable[BaseFeature] | None = Field(
        default=None,
        title="Features",
        description="Features of the dataset",
        example=[
            {
                "feature_name": "transaction_amount",
                "feature_description": "Amount of the transaction",
                "feature_type": "numeric",
                "feature_sub_type": "decimal",
                "feature_rank": 1,
                "is_target": False
            }
        ]
    )

class Guideline(MullyBase):
    """_summary_: Class for guidelines"""
    dataset_id: str = Field(
        ...,
        title="Dataset ID",
        description="ID of the dataset",
        example="123e4567-e89b-12d3-a456-426614174000",
    )
    dataset_goal: str = Field(
        ...,
        title="Dataset Goal",
        description="Goal of the dataset",
        example="To predict ......., To Build ......., To Test .......",
    )
    title: str = Field(
        ...,
        title="Title",
        description="Title of the guideline",
        example="Guideline 1"
    )
    description: str = Field(
        ...,
        title="Description",
        description="Description of the guideline"
    )
    category: MetricType = Field(
        ...,
        title="Category",
        description="Category of the guideline",
        example=MetricType.get_values()
    )
    source: Source = Field(
        ...,
        title="Source",
        description="Source of the guideline",
        example=[
            {
                "name": "source_name",
                "url": "https://www.example.com"
            }
        ]
    )
    rule: str = Field(
        ...,
        title="Rule",
        description="Rule of the guideline",
        example="Rule 1"
    )
    feature_name: str | None = Field(
        None,
        title="Feature Name",
        description="Name of the feature",
        example="transaction_amount"
    )

class DatasetRequest(GuidelineRequest):
    """_summary_: Base Model for Creating Dataset Requests"""
    dataset_name: str | None = Field(
        default=None,
        title="Dataset Name",
        description="Name of the dataset",
        example="iris"
    )
    dataset_sample_size: int = Field(
        default=100,
        title="Dataset Sample Size",
        description="Size of the dataset sample",
        example=100,
        min=100,
        max=1000,
    )
    dataset_output_format: OUTPUT_FORMAT | None = Field(
        default=None,
        title="File Output Format",
        description="Output format for the dataset",
        example="csv",
    )
    dataset_guidelines: Iterable[Guideline] | None = Field(
        [],
        title="Guidelines",
        description="Guidelines for the dataset",
        example=[
            {
                "title": "Guideline 1",
                "description": "Description of the guideline",
                "category": "data privacy",
                "source": {
                    "name": "source_name",
                    "url": "https://www.example.com"
                },
                "rule": "Rule 1",
                "feature_name": "transaction_amount"
            }
        ]
    )

    @model_validator(mode="after")
    def validate_output_format(self) -> Self:
        """_summary_: Validate Output Format"""
        if self.dataset_output_format:
            if not (
                isinstance(self.dataset_output_format, TabularOutputFormat) and
                self.dataset_type == DatasetType.TABULAR
            ):
                raise ValueError(f"""
                        Output format must be tabular for tabular datasets.
                        Select one of {', '.join(TabularOutputFormat.get_values())}
                    """)
            if not (
                isinstance(self.dataset_output_format, DatabaseOutputFormat) and
                self.dataset_type == DatasetType.DATABASE
            ):
                raise ValueError(f"""
                        Output format must be database for database datasets.
                        Select one of {', '.join(DatabaseOutputFormat.get_values())}
                    """)
            if not (
                isinstance(self.dataset_output_format, DocumentOutputFormat) and
                self.dataset_type == DatasetType.DOCUMENT
            ):
                raise ValueError(f"""
                        Output format must be document for document datasets.
                        Select one of {', '.join(DocumentOutputFormat.get_values())}
                    """)
            if not (
                isinstance(self.dataset_output_format, AudioOutputFormat) and
                self.dataset_type == DatasetType.AUDIO
            ):
                raise ValueError(f"""
                        Output format must be audio for audio datasets.
                        Select one of {', '.join(AudioOutputFormat.get_values())}
                    """)
            if not (
                isinstance(self.dataset_output_format, ImageOutputFormat) and
                self.dataset_type == DatasetType.IMAGE
            ):
                raise ValueError(f"""
                        Output format must be image for image datasets.
                        Select one of {', '.join(ImageOutputFormat.get_values())}
                    """)
            if not (
                isinstance(self.dataset_output_format, VideoOutputFormat) and
                self.dataset_type == DatasetType.VIDEO
            ):
                raise ValueError(f"""
                        Output format must be video for video datasets.
                        Select one of {', '.join(VideoOutputFormat.get_values())}
                    """)
        return self


class BaseDataset(DatasetRequest):
    """_summary_: _description_"""

    dataset_files: list[BytesIO] | list[dict] | None = Field(
        default=None,
        title="Dataset File or Object",
        description="File of the dataset",
        example="file.csv"
    )


class EvaluationRequest(BaseDataset):
    """_summary_: Request Class For Validating Datasets"""
    evaluation_model: PROVIDER = Field(
        ...,
        title="Evaluation Model",
        description="Model for the evaluation",
        example=GenerationProvider.get_values(),
    )
    evaluation_metrics : list[MetricType] | None = Field(
        default=None,
        title="Evaluation Metric",
        description="Metric for the evaluation",
        example=MetricType.get_values(),
    )
    dataset_files: list[BytesIO] | list[dict] | None = Field(
        default=None,
        title="Dataset File or Object",
        description="File of the dataset",
        example="file.csv"
    )


class Report(MullyBase):
    """_summary_: Base Model for Reports"""

    dataset_id: str = Field(
        ...,
        title="Dataset ID",
        description="ID of the dataset",
        example="123e4567-e89b-12d3-a456-426614174000",
    )
    model_name: str = Field(
        ...,
        title="Model Name",
        description="Name of the model, that was evaluated",
        example="model_name",
    )
    guidelines: Iterable[Guideline] = Field(
        ...,
        title="Guidelines",
        description="Guidelines for the dataset",
    )
    violations: Iterable[Violation] = Field(
        ...,
        title="Violations",
        description="Violations for the dataset",
    )
    privacy: float | None = Field(
        default=None,
        title="Privacy",
        description="Privacy score of the dataset",
        example=0.9,
    )
    fairness: float | None = Field(
        default=None,
        title="Fairness",
        description="Fairness score of the dataset",
        example=0.9,
    )
    accuracy: float | None = Field(
        default=None,
        title="Accuracy",
        description="Accuracy score of the dataset",
        example=0.9,
    )
    completeness: float | None = Field(
        default=None,
        title="Completeness",
        description="Completeness score of the dataset",
        example=0.9,
    )
    interpretability: float | None = Field(
        default=None,
        title="Interpretability",
        description="Interpretability score of the dataset",
        example=0.9,
    )
    robustness: float | None = Field(
        default=None,
        title="Robustness",
        description="Robustness score of the dataset",
        example=0.9,
    )
    transparency: float | None = Field(
        default=None,
        title="Transparency",
        description="Transparency score of the dataset",
        example=0.9,
    )
