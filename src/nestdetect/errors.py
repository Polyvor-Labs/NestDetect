class NestDetectError(Exception):
    pass


class DatasetValidationError(NestDetectError):
    pass


class DependencyError(NestDetectError):
    pass
