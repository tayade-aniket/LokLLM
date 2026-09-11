class InsufficientMemoryError(RuntimeError):
    pass


class ModelLoadError(RuntimeError):
    pass


class AdapterError(RuntimeError):
    pass


class DataGenerationError(RuntimeError):
    pass


class FederatedError(RuntimeError):
    pass


class PrivacyConfigError(ValueError):
    pass


class BenchmarkError(RuntimeError):
    pass
