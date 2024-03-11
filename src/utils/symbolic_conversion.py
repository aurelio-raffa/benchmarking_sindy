import re


def prepare_for_sympy(equation: str):
    """Adapts a SINDy equation to be used with SymPy.
    """
    equation = equation.replace('+ -', '- ').replace('^', '**')
    equation = re.sub(r'([a-zA-Z]+)\s([a-zA-Z]+)', r'\1 * \2', equation)
    return re.sub(r'(\d+\.\d+)\s([a-zA-Z]+)', r'\1 * \2', equation)
