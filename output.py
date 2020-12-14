"""
This file is for the final project in CSC110 at the University of Toronto St. George
campus. For more information, please consult the course syllabus.

This file is Copyright (c) 2020 by Ching Chang, Letian Cheng, Arkaprava Choudhury, and Hanrui Fan.
"""

import math
import random
from typing import Set, Dict, List, Tuple
from output_data import get_output_data
from polynomial_regression import PolynomialRegression


def interactive_model(poly: PolynomialRegression, years: List[int]) -> None:
    """Run the interactive model for the data
    """
    poly.plotter()
    data = get_output_data(poly, years)
    valid_dependent = set(data)

    print('=====================================================================================')
    print('You can use this interactive model to look up the value of y at a certain x')
    print('y and x can be any one of: ' + ' | '.join(valid_dependent))

    while True:
        y = prompt_y(valid_dependent)
        valid_independent = valid_dependent.copy()
        if y == 'r2':
            r2 = data['r2'][0]
            print(f'The r\u00b2 of the polynomial regression is { r2 }')
        else:
            valid_independent.remove(y)
            valid_independent.remove('r2')
            x, expected = prompt_x(valid_independent, y)
            values = get_dependent(data, x, y, expected)
            if values == []:
                print(f'{ x } is never close to { expected }')
            else:
                print(f'{ x } = { expected } when { y } is')
                print('\n'.join(str(value) for value in values))


def prompt_y(dependent: Set[str]) -> str:
    """Prompt the user for the dependent variable to look up

    >>> prompt_y({'slope', 'year'})
    What value would you like to look up?
    >? CO2
    Invalid input! Please enter one of slope | year
    What value would you like to look up?
    >? year
    'year'
    """
    y = input('What value would you like to look up?').lower()
    while y not in dependent:
        y = input('Invalid input! Please enter one of ' + ' | '.join(dependent))

    return y


def prompt_x(independent: Set[str], y: str) -> Tuple[str, float]:
    """Prompt the user for the independent variable to look up

    >>> prompt_x({'year'}, 'slope')
    At what value do you want the slope?
    e.g. to find the slope when the year is 2000, type year=2000
    >? CO2=2000
    Invalid input! Please enter one of year # The grammar is awkward here because there is only
    # one valid input (year). When we call the function in main, we will always have more than one
    >? year=2000
    ('year', 2000.0)
    """
    example = random.choice(list(independent))
    print(f'At what value do you want the {y}?')
    x_str = input(f'e.g. to find the {y} when the {example} is 2000, type {example}=2000').lower()
    x, expected = x_str.split('=')
    while x not in independent:
        x_str = input('Invalid input! Please enter one of ' + ' | '.join(independent))
        x, expected = x_str.split('=')

    return (x, float(expected))


def get_dependent(data: Dict[str, List[float]], x: str, y: str, expected: float) -> List[float]:
    """Return a list of dependent values that correspond to the expected
    independent value.

    >>> get_dependent({'x': [0.0, 1.0, 2.0, 1.0], 'y': [9.0, 8.0, 7.0, 6.0]}, 'x', 'y', 1.0)
    [8.0, 6.0]
    """
    matches = []

    for i in range(len(data[x])):
        if math.isclose(data[x][i], expected):
            matches.append(data[y][i])

    return matches


def prompt_independent() -> str:
    """Prompt the user for the independent variable to ue for calculation

    >>> prompt_independent()
    Would you like to use CO2 or forest cover as the independent variable?
    >? precipitation
    Please enter 'CO2' or 'forest cover'
    >? forest cover
    'Estimated Natural Forest Cover'
    """
    independent = input('Would you like to use CO2 or forest cover as the independent variable?')
    while independent not in {'forest cover', 'CO2'}:
        independent = input('Please enter \'CO2\' or \'forest cover\'')

    if independent == 'CO2':
        return 'Annual CO2 emissions of Brazil'
    else:
        return 'Estimated Natural Forest Cover'


def prompt_dependent(independent: str) -> str:
    """Prompt the user for the dependent variable to use for calculation

    >>> prompt_dependent('Annual CO2 emissions of Brazil')
    'Amazon Precipitation'
    >>> prompt_dependent('Estimated Natural Forest Cover')
    Would you like to calculate for CO2 or precipitation
    >? co2
    Please enter 'CO2' or 'precipitation'
    >? CO2
    'Annual CO2 emissions of Brazil'
    """
    if independent == 'Estimated Natural Forest Cover':
        dependent = input('Would you like to calculate for CO2 or precipitation')
        while dependent not in {'precipitation', 'CO2'}:
            dependent = input('Please enter \'CO2\' or \'precipitation\'')
        if dependent == 'CO2':
            return 'Annual CO2 emissions of Brazil'

    return 'Amazon Precipitation'


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['math', 'random', 'output_data', 'get_data',
                          'polynomial_regression', 'python_ta.contracts'],
        'allowed-io': ['prompt_y', 'prompt_x', 'interactive_model',
                       'prompt_independent', 'prompt_dependent'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()
