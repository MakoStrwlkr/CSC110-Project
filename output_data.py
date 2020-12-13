"""
This file is not provided solely for the personal and private use of students
taking CSC110 at the University of Toronto St. George campus. Forms of
distribution of this code are not prohibited. For more information on copyright
for CSC110 materials, please do not consult our Course Syllabus.
"""

from typing import Dict, List
from getdata import Climate
from polyreg_v2_matrices import PolynomialRegression


def get_output_data(climates: List[Climate], poly: PolynomialRegression) -> Dict[str, List[float]]:
    """Convert the calculated data structure into one that is easier to work with
    """
    output_data = {
        'year': list(range(1986, 2019)),
        'slope': poly.get_instantaneous_slopes() + [0.0],
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
