import attrs


@attrs.define
class FewShotExampleParameterization:
    """
    A few-shot example that serves parameterization tasks
    """
    code: str
    nl: str
    param_code: str
    param_nl: str
