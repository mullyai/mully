from enum import Enum

class MullyEnum(str, Enum):
    """_summary_: Base class for Mully Enums."""

    @classmethod
    def get_choices(cls):
        """_summary_: Get choices for the enum.
        _description_: This method returns the choices for the enum.
        """
        return [(item.name, item.value) for item in cls]

    @classmethod
    def get_values(cls):
        """_summary_: Get values for the enum.
        _description_: This method returns the values for the enum.
        """
        return [item.value for item in cls]


class TaskType(MullyEnum):
    """_summary_: Enum for Mully job type.
    _description_: This enum is used to store the Mully job type.
    """
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    GENERATION = "generation"

class DatasetType(MullyEnum):
    """_summary_: Enum for dataset type.
    _description_: This enum is used to store the dataset type.
    """
    TABULAR = "tabular"
    DATABASE = "database"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"

class DatasetSector(MullyEnum):
    """_summary_: Enum for Mully Agent Domain."""

    # TODO: Add Business Domains inline with EU, UK, US, AUS, CANADA AI Act
    HEALTH = "health"
    FINANCE = "finance"
    COPORATE_LAW = "corporate law"
    CRIMINAL_LAW = "criminal law"
    EDUCATION = "education"
    ANY = "any"

class DatasetCountry(MullyEnum):
    """_summary_: Enum for Mully DatasetCountry"""
    UNITED_KINGDOM = "united kingdom"
    UNITED_STATES = "united states"
    CANADA = "canada"
    AUSTRALIA = "australia"
    GERMANY = "germany"
    FRANCE = "france"
    ITALY = "italy"
    SPAIN = "spain"
    JAPAN = "japan"
    CHINA = "china"
    ANY = "any country"

class DatasetLanguage(MullyEnum):
    """_summary_: Enum for Mully DatasetLanguage"""
    ENGLISH = "english"
    FRENCH = "french"
    GERMAN = "german"
    SPANISH = "spanish"
    ITALIAN = "italian"
    JAPANESE = "japanese"
    CHINESE = "chinese"

class MetricType(MullyEnum):
    """_summary_: Enum for Mully Guide Category"""
    PRIVACY = "privacy"
    SECURITY = "security"
    ROBUSTNESS = "robustness"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    FAIRNESS = "fairness"
    MINIMIZATION = "minimization"

class GenerationProvider(MullyEnum):
    """_summary_: Enum for Mully LLM."""
    OPENAI = "openai"
    '''GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"'''

class ClassificationProvider(MullyEnum):
    """_summary_: Enum for Mully Classification Provider."""
    CATBOOST = "catboost"
    XGBOOST = "xgboost"
    RANDOMFOREST = "randomforest"

class ClusteringProvider(MullyEnum):
    """_summary_: Enum for Mully Clustering Provider."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"

class RegressionProvider(MullyEnum):
    """_summary_: Enum for Mully Regression Provider."""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    RIDGE = "ridge"
    LASSO = "lasso"

class TabularOutputFormat(MullyEnum):
    """_summary_: Enum for Mully Text Output Format"""
    CSV = "csv"
    JSON = "json"

class DatabaseOutputFormat(MullyEnum):
    """_summary_: Enum for Mully Database Output Format"""
    SQLITE = "sqlite"
    JSON = "json"
    XLXS = "xlxs"

class DocumentOutputFormat(MullyEnum):
    """_summary_: Enum for Mully Text Output Format"""
    TXT = "txt"
    DOCX = "docx"
    MARKDOWN = "markdown"
    PDF = "pdf"

class AudioOutputFormat(MullyEnum):
    """_summary_: Enum for Mully Audio Output Format"""
    MP3 = "mp3"
    WAV = "wav"
    MIDI = "midi"
    
class ImageOutputFormat(MullyEnum):
    """_summary_: Enum for Mully Image Output Format"""
    PNG = "png"
    JPG = "jpg"
    GIF = "gif"
    
class VideoOutputFormat(MullyEnum):
    """_summary_: Enum for Mully Video Output Format"""
    MP4 = "mp4"
    MOV = "mov"
    