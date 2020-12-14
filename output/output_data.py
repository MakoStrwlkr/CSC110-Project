"""
This file is for the final project in CSC110 at the University of Toronto St. George
campus. For more information, please consult the course syllabus.

This file is Copyright (c) 2020 by Ching Chang, Letian Cheng, Arkaprava Choudhury, and Hanrui Fan.
"""

from typing import Dict, List, Any
from polynomial_regression import PolynomialRegression


def get_output_data(poly: PolynomialRegression, years: List[int]) -> Dict[str, List[Any]]:
    """Convert the calculated data structure into one that is easier to work with
    """
    output_data = {
        'slope': poly.get_instantaneous_slopes(),
        'error': poly.error_values,
        'r2': [poly.find_r_squared()]
    }

    if poly.x_var == 'Annual CO2 emissions of Brazil':
        output_data['CO2'] = poly.x_values
        output_data['precipitation'] = poly.y_values
    elif poly.y_var == 'Annual CO2 emissions of Brazil':
        output_data['tree area'] = poly.x_values
        output_data['CO2'] = poly.y_values
    else:
        output_data['tree area'] = poly.x_values
        output_data['precipitation'] = poly.y_values

    output_data['year'] = years

    return output_data


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['typing', 'get_data', 'polynomial_regression', 'python_ta.contracts'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()
