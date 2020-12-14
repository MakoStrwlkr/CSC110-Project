""" CSC110 Course Project
Data Analysis using Polynomial regression implemented using matrices through
the ordinary least squares estimator.

Written by Arkaprava Choudhury,
In partnership with Ching Chang, Letian Cheng, and Hanrui Fan.

All work presented here written by Arkaprava Choudhury, et. al, and no piece of code has been
referenced from any sources on the Internet. The authors reserve all rights to replicate this
work, and no one, apart from the graders of this project and the instructors of CSC110 can
modify / use this code as their own.

This file is Copyright (c) 2020 by Ching Chang, Letian Cheng, Arkaprava Choudhury, and Hanrui Fan.
"""

import math
from typing import List, Any, Dict
import numpy as np
import matplotlib.pyplot as plt


############################################################################################
# Functions for matrices
############################################################################################

def transpose_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """Return the transpose of the input matrix.

    Note: will work for only real inputs, so cannot extend this to
    complex polynomials.

    Preconditions:
      - matrix must be in the form returned by make_matrix.

    >>> transpose_matrix([[1, 1, 1], [1, 2, 4], [1, 3, 9]])
    [[1, 1, 1], [1, 2, 3], [1, 4, 9]]
    """
    transpose = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

    return transpose


def matrix_multiplication_small(matrix_1: List[List[float]], matrix_2: List[List[float]])\
        -> List[List[float]]:
    """Return the matrix matrix_1 * matrix_2, where * represents matrix multiplication.

    Preconditions:
      - len(matrix_1[0]) == len(matrix_2)

    >>> mat_1 = [[1], [1]]
    >>> mat_2 = [[1, 1]]
    >>> matrix_multiplication_small(mat_1, mat_2)
    [[1, 1], [1, 1]]
    """
    # Note: This is not the most efficient way to do matrix multiplication.
    #       While it can certainly be made more efficient, the improvements for our purposes is
    #       negligible since we are only concerned with small degrees, and so, this function
    #       suffices.

    multiple = [[0 for _ in range(len(matrix_2[0]))] for _ in range(len(matrix_1))]

    for i in range(len(matrix_1)):
        for j in range(len(matrix_2[0])):
            for k in range(len(matrix_2)):
                multiple[i][j] += matrix_1[i][k] * matrix_2[k][j]

    return multiple


def matrix_inverse_numpy(matrix: List[List[float]]) -> Any:
    """Return the matrix inverse of the input matrix.

    Preconditions:
      - len(matrix) == len(matrix[0])
      - matrix is invertible.
    """
    # Since I could not find any efficient algorithm for finding the inverse
    # of a matrix, I used numpy, and a data class in numpy called array to
    # optimize the process.
    matrix_inv_numpy = np.linalg.inv(matrix)
    return matrix_inv_numpy.tolist()


def find_coefficients(x_data: List[float], y_data: List[float], degree: int) -> List[List[float]]:
    """Return the coefficients of the estimate polynomial that approximates
    the given data, as per the ordinary least squares estimator function.

    This method was chosen, as it minimizes the sum of the squares of residuals,
    and consequently, it also minimizes the variance, as proven by the Gauss-Markov theorem.

    The list of estimated coefficients contains the coefficients of 1, x, x^2, ...
    till x^(degree).

    This list is calculated using the formula:
    beta = (X^T * X)^(-1) * X^T * y
    where X represents the matrix of powers of the x values, and y represents a list
    of the y values.

    For more information on the correctness of this algorithm, visit these Wikipedia pages:
    https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem
    https://en.wikipedia.org/wiki/Polynomial_regression

    Preconditions:
      - 0 < degree < len(x_data)
      - all(x_data[i] != x_data[j] for i in range(len(x_data))
            for j in range(len(x_data)) if i != j)
    """
    x_matrix = make_matrix(x_data, degree)
    y_matrix = [[y] for y in y_data]

    x_transpose = transpose_matrix(x_matrix)
    multiply = matrix_multiplication_small(x_transpose, x_matrix)

    inv_matrix = matrix_inverse_numpy(multiply)
    inv_multiply_with_transpose = matrix_multiplication_small(inv_matrix, x_transpose)

    beta = matrix_multiplication_small(inv_multiply_with_transpose, y_matrix)

    return beta


############################################################################################
# This function is used to make the matrix X in the matrix equation specified above.
############################################################################################

def make_matrix(points: List[float], degree: int) -> List[List[float]]:
    """Return a nested list representation of a matrix consisting of the basis
    elements of the polynomial of degree n, evaluated at each of the points.

    In other words, each row consists of 1, x, x^2, ..., x^n, where n is the degree,
    and x is a value in points.

    Preconditions:
      - degree < len(points)

    >>> make_matrix([1, 2, 3], 2)
    [[1, 1, 1], [1, 2, 4], [1, 3, 9]]
    """
    matrix = []

    for point in points:
        row = [point ** index for index in range(degree + 1)]
        matrix.append(row)

    return matrix


############################################################################################
# Polynomial class, and defining polynomial, and derivative, which will be useful
# in the output section of this project.
############################################################################################


class PolynomialAbstract:
    """A representation of a polynomial function.

    This is an abstract class. Only subclasses should be instantiated.
    """
    coefficients: List[float]

    def __call__(self, value: float) -> float:
        """Call the polynomial.

        This makes it easier to evaluate the polynomial function at any value.
        """
        raise NotImplementedError


class Polynomial(PolynomialAbstract):
    """A polynomial function class to represent a polynomial in Python.

    This was defined to allow for defining the derivative of a polynomial
    as used in the output section of this project.

    Public Instance Attributes:
      - coefficients: the coefficients of the terms of the polynomial, from
                      lowest order to highest.
    """
    coefficients: List[float]

    def __init__(self, coefficients: List[float]) -> None:
        """ Initialize the polynomial when inputted a list of coefficients
         in the order a_0, a_1, ..., a_{n-1}, a_n, i.e.,
         coefficients[i] represents the coefficient of the term with x^i
         for the polynomial.

         Preconditions:
           - coefficients != []

        """
        self.coefficients = coefficients

    def __call__(self, value: float) -> float:
        """A method to call the polynomial. This makes it easier to evaluate the
        polynomial function at any value.

        Preconditions:
          - self has already been initialized (i.e., init has already run).
        """
        evaluation = 0
        for index, coefficient in enumerate(self.coefficients):
            evaluation += coefficient * value ** index
        return evaluation


############################################################################################
# PolynomialRegression class. This is the most important part of this file.
############################################################################################

class PolynomialRegression(PolynomialAbstract):
    """ The polynomial regression model of the data.

    Public Instance Attributes:
      - coefficients: the coefficients of the terms of the polynomial, from
                      lowest order to highest.
      - x_values: the list of values for the independent (X) variable
      - y_values: the list of values for the dependent (Y) variable
      - x_var: the name of the independent (X) variable
      - y_var: the name of the dependent (Y) variable
      - error_values: the list of error values obtained from the difference of the
                      a y-value and the polynomial model evaluated at the corresponding
                      x-value
      - r_squared: the coefficient of determination for the polynomial model generated

    Representation invariants:
      - self.coefficients != []
      - self.x_values != []
      - self.y_values != []
      - len(self.x_values) = len(self.y_values)
      - len(self.coefficients) <= len(self.x_values)
      - len(self.error_values) = len(self.x_values)
      - 0 <= self.r_squared <= 1

    """
    coefficients: List[float]
    x_values: List[float]
    y_values: List[float]
    x_var: str
    y_var: str
    error_values: List[float]
    r_squared: float

    def __init__(self, x_dict: Dict[str, List[float]], y_dict: Dict[str, List[float]], precision: int) -> None:
        """Initialize the polynomial regression model of specified degree
        generated by input data.

        precision refers to the maximum number of decimals in the returned values.
        This is done to avoid floating point error.

        self.coefficients is in the form [a_0, a_1, ..., a_(degree), a_(degree)]

        More preconditions are as specified in the respective function docstrings.

        Preconditions:
          - 0 <= precision
          - len(x_dict) == 1
          - len(y_dict) == 1
        """

        # initialize the x-var and y-var for the regression model
        self.x_var = list(x_dict)[0]
        self.y_var = list(y_dict)[0]

        # initialize the x- and y-values in the regression model
        self.x_values = x_dict[self.x_var]
        self.y_values = y_dict[self.y_var]

        # find the coefficient matrix (column matrix) for the input data
        beta = find_coefficients(self.x_values, self.y_values, 2)

        # to avoid floating point error, round it to specified number of decimals
        coefficients = [round(b[0], precision) for b in beta]

        self.coefficients = coefficients

        self.error_values = [round(self.y_values[i] - self(self.x_values[i]), precision)
                             for i in range(len(self.x_values))]
        self.r_squared = round(self.find_r_squared(), precision)

        mean_error = expected_value(self.error_values)

        self.coefficients[0] += mean_error

    def __call__(self, value: Any) -> float:
        """A method to call the polynomial. This makes it easier to evaluate the
        polynomial function at any value.

        >>> polynomial = PolynomialRegression({'year': [1, 2, 3]}, {'precip': [1, 4, 9]}, 5)
        >>> math.isclose(polynomial(4), 16.0)
        True
        """
        # Sorry, this code is repeated from above, but when I tried to make __call__
        # a method in the abstract class, I got errors from PyCharm.
        # although the code smells a bit, this is manageable.

        evaluation = 0
        for index, coefficient in enumerate(self.coefficients):
            evaluation += coefficient * value ** index
        return evaluation

    def plotter(self) -> None:
        """Plot a simple graph of the polynomial, as well as a scatter graph
        of values in the two lists.
        """
        x = np.linspace(0.5 * min(self.x_values), 1.25 * max(self.x_values),
                        100, endpoint=True)
        f = self(x)
        plt.plot(x, f)

        plt.style.use('seaborn')

        max_val = max(self.y_values)
        min_val = min(self.y_values)

        colour_gradient = 6 / (max_val - min_val)
        colour_intercept = 4 - (colour_gradient * min_val)

        colours = [colour_gradient * y_val + colour_intercept for y_val in self.y_values]

        plt.scatter(self.x_values, self.y_values, c=colours, cmap='Blues',
                    edgecolor='black', linewidth=1, alpha=0.75)

        cbar = plt.colorbar()
        cbar.set_label(self.y_var)

        plt.xlabel(self.x_var)
        plt.ylabel(self.y_var)

        plt.tight_layout()

        plt.show()

    def extreme_absolute_error(self) -> Dict[str, float]:
        """Return a dictionary of minimum absolute error: value
        and maximum absolute error: value.

        Preconditions:
          - self.coefficients != []
        """
        absolute_errors = [abs(error) for error in self.error_values]
        return {'min absolute error': min(absolute_errors),
                'max absolute error': max(absolute_errors)}

    def find_r_squared(self) -> float:
        """Return the coefficient of determination for the polynomial regression model
        """
        mean_y = expected_value(self.y_values)
        total_sum = sum((y - mean_y) ** 2 for y in self.y_values)
        residual_sum = sum((self.y_values[i] - self(self.x_values[i])) ** 2
                           for i in range(len(self.x_values)))

        return 1 - (residual_sum / total_sum)

    def covariance_with_polynomial(self) -> float:
        """Return the covariance of the y_values in the data, and the polynomial model.
        """

        product_list = [self.y_values[i] * self(self.x_values[i])
                        for i in range(len(self.x_values))]
        expected_product = expected_value(product_list)

        expected_polynomial = expected_value([self(self.x_values[i])
                                              for i in range(len(self.x_values))])

        return expected_product / (expected_value(self.y_values) * expected_polynomial)

    def correlation_of_data(self) -> float:
        """Return the correlation of the y_values in the data, with the x_values.
        """

        product_list = [self.y_values[i] * self.x_values[i]
                        for i in range(len(self.x_values))]
        expected_product = expected_value(product_list)

        expected_x_expected_y = expected_value(self.y_values) * expected_value(self.x_values)

        numerator = expected_product / expected_x_expected_y
        denominator = standard_deviation(self.y_values) * standard_deviation(self.x_values)

        return numerator / denominator

    def differentiate(self) -> Polynomial:
        """Differentiate the given polynomial.

        Returns the derivative of the polynomial.

        >>> polynomial = PolynomialRegression({'year': [1, 2, 3]}, {'precip': [1, 4, 9]}, 2, 5)
        >>> diff = polynomial.differentiate()
        >>> diff.coefficients == [0, 2]
        True
        """
        derivative = self.coefficients.copy()
        for i in range(len(derivative)):
            derivative[i] *= i

        derivative.pop(0)
        return Polynomial(derivative)

    def get_instantaneous_slopes(self) -> List[float]:
        """Get the instantaneous slope of the regression line at each x value

        >>> poly = PolynomialRegression({'year': [1, 2, 3]}, {'precip': [1, 4, 9]}, 2, 5)
        >>> poly.get_instantaneous_slopes()
        [2.0, 4.0, 6.0]
        """
        slopes = []
        derivative = self.differentiate()
        coefficients = derivative.coefficients
        for x in self.x_values:
            slope = 0
            for i in range(len(coefficients)):
                slope += coefficients[i] * x ** i
            slopes.append(slope)

        return slopes


############################################################################################
# Some helper functions to help with error calculations.
############################################################################################


def expected_value(values: List[float]) -> float:
    """Return the expected value of the input list

    >>> expected_value([1, 2, 3])
    2.0
    """
    return sum(values) / len(values)


def standard_deviation(values: List[float]) -> float:
    """Return the standard deviation of the input list

    >>> standard_deviation([1, 1, 1])
    0.0
    """
    mean_val = expected_value(values)
    sum_of_squares = sum([(val - mean_val) ** 2 for val in values])
    return math.sqrt(sum_of_squares / (len(values) - 1))


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=True)

    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['numpy', 'matplotlib.pyplot', 'typing', 'math'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
