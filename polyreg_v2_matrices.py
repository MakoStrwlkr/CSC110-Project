""" CSC110 Course Project
Data Analysis using Polynomial regression implemented using matrices through
the ordinary least squares estimator.

Written by Arkaprava Choudhury,
In partnership with Ching Chang, Letian Cheng, and Hanrui Fan.

All work presented here written by Arkaprava Choudhury, et. al, and no piece of code has been
referenced from any sources on the Internet. The authors reserve all rights to replicate this
work, and no one, apart from the graders of this project and the instructors of CSC110 can
modify / use this code as their own.

This file is Copyright (c) by Ching Chang, Letian Cheng, Arkaprava Choudhury, and Hanrui Fan.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any, Dict
import math


############################################################################################
# Input function
# takes in a list of co-ordinates and splits them up.
############################################################################################


# We changed the way the PolynomialRegression class initializes values so we didn't use this.
# def split_coordinates(coordinates: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
#     """Return a tuple of the x-values and y-values.
#
#     Preconditions:
#       - coordinates != []
#
#     >>> split_coordinates([(1, 1), (2, 4), (3, 9)])
#     ([1, 2, 3], [1, 4, 9])
#     """
#     points = sorted(coordinates)
#     return ([x[0] for x in points], [y[1] for y in points])


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


############################################################################################
# Initially, I had planned on using numpy's matrix multiplication function, which is much more
# efficient than the naive approach I used.
# Although I know how to make this implementation faster using the module tensorflow, the exact
# algorithm used is beyond the scope of the course or any other math courses I have taken, and
# as such, I was not fully convinced of its correctness, and so, did not use it.
# For the purposes of this project, a Theta(m * n * k) algorithm is sufficient, where matrix_1 has
# dimension m * n, and matrix_2 had dimension n * k; this is because, in polynomial regression,
# we usually don't take high-order polynomials.
############################################################################################

# def matrix_multiplication_numpy(matrix_1: List[List[float]], matrix_2: List[List[float]]) \
#         -> List[List[float]]:
#     """Return the matrix matrix_1 * matrix_2, where * represents matrix multiplication.
#
#     Preconditions:
#       - len(matrix_1[0]) == len(matrix_2)
#
#     >>> mat_1 = [[1], [1]]
#     >>> mat_2 = [[1, 1]]
#     >>> matrix_multiplication_numpy(mat_1, mat_2)
#     [[1, 1], [1, 1]]
#     """
#     m1 = np.array(matrix_1)
#     m2 = np.array(matrix_2)
#     multiple = np.matmul(m1, m2)
#     return multiple.tolist()


def matrix_multiplication_small(matrix_1: List[List[float]], matrix_2: List[List[float]])\
        -> List[List[float]]:
    """Return the matrix matrix_1 * matrix_2, where * represents matrix multiplication.

    Note: This is not the most efficient way to do matrix multiplication.
    While it can certainly be made more efficient, we have not covered the
    required mathematics foundations to do this.

    As such, I am not fully convinced of why that algorithm works, but I have still
    translated it into Python code, and added it to the addendum file
    efficient_functions.py as seen there.

    Preconditions:
      - len(matrix_1[0]) == len(matrix_2)

    >>> mat_1 = [[1], [1]]
    >>> mat_2 = [[1, 1]]
    >>> matrix_multiplication_small(mat_1, mat_2)
    [[1, 1], [1, 1]]
    """
    multiple = [[0 for _ in range(len(matrix_2))] for _ in range(len(matrix_1))]

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

    # mult_beta_x = matrix_multiplication_small(x_matrix, beta)
    #
    # epsilon = [y_matrix[i][0] - mult_beta_x[i][0] for i in range(len(y_matrix))]

    return beta


############################################################################################
# Originally, I had planned on using a custom class Matrix represent the matrices used in this
# file, but I decided that using List[List[float]] type annotations were a better way of
# representing the different shapes of matrices used (column matrix, rectangular matrix, and
# square matrix).
# Or else, I could have used the numpy array data class, but that required me to repeatedly call
# the tolist() function in my own functions, so I did not choose that.
############################################################################################

# class Matrix:
#     """An abstract data class for a nested list representation of a matrix.
#     All subclasses support matrix transpose, matrix inverse, matrix multiplication.
#
#     """
#
#     ...
#
#
# class XMatrix(Matrix):
#     """Matrix for storing the powers of x.
#
#     Public instance attributes:
#       - matrix: nested list representation of a matrix
#     """
#     matrix: List[List[float]]
#
#     def __init__(self, points: List[float], degree: int) -> None:
#         """Returns a nested list representation of a matrix consisting of the basis
#         elements of the polynomial of degree n, evaluated at each of the points.
#
#         In other words, each row consists of 1, x, x^2, ..., x^n, where n is the degree,
#         and x is a value in points.
#
#         Preconditions:
#           - degree < len(points)
#         """
#         self.matrix = make_matrix(points, degree)

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

    def __call__(self, value) -> float:
        """A method to call the polynomial. This makes it easier to evaluate the
        polynomial function at any value.
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

    def __init__(self, coefficients) -> None:
        """ Initialize the polynomial when inputted a list of coefficients
         in the order a_0, a_1, ..., a_{n-1}, a_n, i.e.,
         coefficients[i] represents the coefficient of the term with x^i
         for the polynomial.

         Preconditions:
           - coefficients != []

        """
        self.coefficients = coefficients

    def __call__(self, value) -> float:
        """A method to call the polynomial. This makes it easier to evaluate the
        polynomial function at any value.
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
    error_values: List[float]
    r_squared: float

    def __init__(self, x_values: List[float], y_values: List[float], degree: int, precision: int) \
            -> None:
        """Initialize the polynomial regression model of specified degree
        generated by input data.

        coefficients is in the form [a_0, a_1, ..., a_(degree), a_(degree)]

        Preconditions are as specified in the respective function docstrings.
        """
        self.x_values = x_values
        self.y_values = y_values

        beta = find_coefficients(x_values, y_values, degree)
        coefficients = [round(b[0], precision) for b in beta]

        self.coefficients = coefficients

        self.error_values = [round(y_values[i] - self(x_values[i]), precision)
                             for i in range(len(x_values))]
        self.r_squared = round(self.find_r_squared(), precision)

    def __call__(self, value) -> float:
        """A method to call the polynomial. This makes it easier to evaluate the
        polynomial function at any value.
        """
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

        plt.scatter(self.x_values, self.y_values)

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
        total_sum = sum([(y - mean_y) ** 2 for y in self.y_values])
        residual_sum = sum([(self.y_values[i] - self(self.x_values[i])) ** 2
                            for i in range(len(self.x_values))])

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

        Returns a list of coefficients for the derivative of the polynomial.
        """
        derivative = self.coefficients.copy()
        degree = len(derivative) - 1
        for i in range(degree):
            derivative[i] *= degree - i

        return Polynomial(derivative.pop())

    def get_instantaneous_slopes(self) -> List[float]:
        """Get the instantaneous slope of the regression line at each x value
        """
        slopes = []
        derivative = self.differentiate()
        coefficients = derivative.coefficients
        for x in self.x_values:
            for i in range(len(coefficients)):
                slopes.append(coefficients[i] * x ** (len(coefficients) - 1 - i))

        return slopes




############################################################################################
# Some functions to help with error calculations.
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
    0
    """
    mean_val = expected_value(values)
    sum_of_squares = sum([(val - mean_val) ** 2 for val in values])
    return math.sqrt(sum_of_squares / (len(values) - 1))
