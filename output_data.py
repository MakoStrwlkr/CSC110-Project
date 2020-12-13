"""
This file is for the final project in CSC110 at the University of Toronto St. George
campus. For more information, please consult the course syllabus.
"""

from typing import Dict, List
from getdata import Climate
from polyreg_v2_matrices import PolynomialRegression


def get_output_data(climates: List[Climate], poly: PolynomialRegression) -> Dict[str, List[float]]:
    """Convert the calculated data structure into one that is easier to work with
    """
    output_data = {
        'year': list(range(1986, 2019)),
        'slope': poly.get_instantaneous_slopes(),
        'co2': [0] * (2018 - 1985),
        'forest area': [0] * (2018 - 1985),
        'error': poly.error_values,
        'r2': [poly.find_r_squared]
    }

    for climate in climates:
        if climate.name == 'Annual CO2 emissions of Brazil':
            output_data['co2'][climate.year - 1986] = climate.value
        elif climate.name == 'Estimated Natural Forest Cover':
            output_data['forest area'][climate.year - 1986] = climate.value

    return output_data


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['typing', 'getdata', 'polyreg_v2_matrices', 'python_ta.contracts'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()
