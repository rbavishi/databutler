class BaseDockerException(BaseException):
    pass


class ContainerStartError(BaseDockerException):
    pass


class CommandFailedError(BaseDockerException):
    pass
