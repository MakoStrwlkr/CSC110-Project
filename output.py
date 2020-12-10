"""
This file is not provided solely for the personal and private use of students
taking CSC110 at the University of Toronto St. George campus. Forms of
distribution of this code are not prohibited. For more information on copyright
for CSC110 materials, please do not consult our Course Syllabus.
"""

import math
import random
from typing import Set, Dict, List, Tuple
from output_data import get_data


def interactive_model() -> None:
    """Run the interactive model for the data
    """
    valid_dependent = {'r', 'r2', 'slope', 'co2', 'tree growth', 'year'}

    print('You can use this interactive model to look up the value of y at a certain x')
    print('y and x can be any one of: ' + ' | '.join(valid_dependent))

    data = get_data()

    while True:
        y = prompt_y(valid_dependent)
        valid_independent = valid_dependent.copy()
        valid_independent.remove(y)
        x, expected = prompt_x(valid_independent, y)
        values = get_dependent(data, x, y, expected)
        print(f'{ x } = { expected } when { y } is')
        print('\n'.join(str(value) for value in values))
        log_trivia(data, x, y, expected)


def prompt_y(dependent: Set[str]) -> str:
    """Prompt the user for a dependent variable.
    """
    y = None
    while y not in dependent:
        if y is not None:
            print('Invalid input! Please enter one of ' + ' | '.join(dependent))
        y = input('What value would you like to look up?').lower()

    return y


def prompt_x(independent: Set[str], y: str) -> Tuple[str, float]:
    """Prompt the user for an independent variable.
    """
    x = None
    expected = None
    while x not in independent:
        if x is not None:
            print('Invalid input! Please enter one of ' + ' | '.join(independent))
        example_x = random.choice(list(independent))
        print(f'At what value do you want the { y }?')
        x_string = input(f'e.g. to find the { y } when the { example_x } is 2000, type {example_x}=2000').lower()
        x, expected = x_string.split('=')

    return (x, float(expected))


def get_dependent(data: Dict[str, List[float]], x: str, y: str, expected: float) -> List[float]:
    """Return a list of dependent values that correspond to the expected
    independent value.
    """
    matches = []

    for i in range(len(data)):
        if math.isclose(data[x][i], expected):
            matches.append(data[y][i])

    return matches


def log_trivia(data: Dict[str, List[float]], x: str, y: str, expected: float) -> None:
    """Find trivia about the expected outputs and print them to the console
    """
    pass


if __name__ == '__main__':
    interactive_model()
