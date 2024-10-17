class ToolkitException(Exception):
    """Base exception class for the data analysis toolkit."""
    def __init__(self, message, error=None):
        self.message = message
        self.error = error
        super().__init__(self.message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}" + (f" | Original error: {self.error}" if self.error else "")

class ProcessingError(ToolkitException):
    """Exception raised for errors in the data processing pipeline."""
    pass

class AnalysisError(ToolkitException):
    """Exception raised for errors in the data analysis stage."""
    pass

class DatabaseError(ToolkitException):
    """Exception raised for database-related errors."""
    pass

class InputError(ToolkitException):
    """Exception raised for invalid input data."""
    pass

class LanguageError(ToolkitException):
    """Exception raised for language-related errors."""
    pass

class ModelError(ToolkitException):
    """Exception raised for errors related to machine learning models."""
    pass

class TokenizationError(ProcessingError):
    """Exception raised for errors during text tokenization."""
    pass

class LemmatizationError(ProcessingError):
    """Exception raised for errors during lemmatization."""
    pass

class VectorizationError(AnalysisError):
    """Exception raised for errors during text vectorization."""
    pass

class TopicModelingError(AnalysisError):
    """Exception raised for errors during topic modeling."""
    pass

class DatabaseConnectionError(DatabaseError):
    """Exception raised when unable to connect to the database."""
    pass

class DatabaseQueryError(DatabaseError):
    """Exception raised when a database query fails."""
    pass

class UnsupportedLanguageError(LanguageError):
    """Exception raised when an unsupported language is specified."""
    pass

class ModelLoadingError(ModelError):
    """Exception raised when unable to load a machine learning model."""
    pass

class InsufficientDataError(AnalysisError):
    """Exception raised when there's not enough data for analysis."""
    pass

class ConfigurationError(ToolkitException):
    """Exception raised for configuration-related errors."""
    pass

class APIError(ToolkitException):
    """Exception raised for API-related errors."""
    pass

class FileOperationError(ToolkitException):
    """Exception raised for file-related operations."""
    pass