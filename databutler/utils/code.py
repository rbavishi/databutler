import ast

import astunparse


def normalize_code(code: str) -> str:
    """
    Returns a formatting-normalized version of the code by running through a parser and then a code-generator.

    Args:
        code: A string corresponding to the code to normalize

    Returns:
        (str): The normalized code.
    """
    return astunparse.unparse(ast.parse(code)).strip()
