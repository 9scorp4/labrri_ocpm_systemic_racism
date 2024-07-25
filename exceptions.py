class ToolkitException(Exception):
    """Base exception class for the data analysis toolkit."""
    def __init__(self, message, error=None):
        self.message = message
        self.error = error
        super().__init__(self.message)

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