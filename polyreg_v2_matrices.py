""" CSC110 Course Project - Data Analysis

Polynomial regression implemented using matrices.

In contrast to v1, where the program used lists to represent polynomials,
this program uses a different approach to finding the polynomial regression model.

A simpler approach is used here, relying more on numpy's matrix feature,
which I plan to rewrite using my own functions, similar to the transpose functions
used in Assignment 4.

The sole limitation of this model lies in calculating the errors; the complexity of
the calculations involved increases with this approach as the number of calculations changes.

Overall, I'd say this method is better, (well, since this has a complete implementation, and
the other doesn't).

"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Any


############################################################################################
# Input function
# takes in a list of co-ordinates and splits them up.
############################################################################################

def split_coordinates(coordinates: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
    """Returns a tuple of the x-values and y-values.

    Preconditions:
      - coordinates != []
    """
    points = sorted(coordinates)
    return ([x[0] for x in points], [y[1] for y in points])


def transpose_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """Returns the transpose of the input matrix.

    Note: will work for only real inputs, so cannot extend this to
    complex polynomials.

    Matrix must be in the form returned by make_matrix.
    """
    transpose = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

    return transpose


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
#     return list(np.multiply(m1, m2))


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
    # I will keep trying to build an efficient self-written code in the next
    # two days, and add what I get to this file.
    matrix_inv_numpy = np.linalg.inv(matrix)
    return matrix_inv_numpy.tolist()


def find_coefficients(x_data: List[float], y_data: List[float], degree: int) -> List[List[float]]:
    """Returns the coefficients of the estimate polynomial that approximates
    the given data.

    The list of estimated coefficients contains the coefficients of 1, x, x^2, ...
    till x^(degree).

    This list is calculated using the formula:
    beta = (X^T * X)^(-1) * X^T * y
    where X represents the matrix of powers of the x values, and y represents a list
    of the y values.

    Preconditions:
      - degree < len(x_data)
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


# class Matrix:
#     """An abstract data class for a nested list representation of a matrix.
#     All subclasses support matrix transpose, matrix inverse, matrix multiplication.
#
#     Public Instance Attributes:
#       - matrix: nested list representation of a matrix.
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


def make_matrix(points: List[float], degree: int) -> List[List[float]]:
    """Returns a nested list representation of a matrix consisting of the basis
    elements of the polynomial of degree n, evaluated at each of the points.

    In other words, each row consists of 1, x, x^2, ..., x^n, where n is the degree,
    and x is a value in points.

    Preconditions:
      - degree < len(points)
    """
    matrix = []

    for point in points:
        row = [point ** index for index in range(degree + 1)]
        matrix.append(row)

    return matrix


class PolynomialRegression:
    """Test case 1. Prototype version 2.0.

    ver 1.1.: Defining polynomial function class to allow callables.
    ver 2.0.: Change code to allow passing list of coordinates, that are changed to
              two lists, and then, call find_coefficients, which are used as the coefficients
              for the polynomial.
              Also, the plotter has been modified slightly, although it can still be improved.

    Public Instance Attributes:
      - coefficients: the coefficients of the terms of the polynomial, from
                      lowest order to highest.
    """
    coefficients: List[float]
    x_values: List[float]
    y_values: List[float]

    def __init__(self, data: List[Tuple[float, float]], degree: int) -> None:
        """

        input: coefficients are in the form a_0, a_1, ..., a_{n-1}, a_n
        """
        x_values, y_values = split_coordinates(data)

        beta = find_coefficients(x_values, y_values, degree)
        coefficients = [b[0] for b in beta]

        self.coefficients = coefficients
        self.x_values = x_values
        self.y_values = y_values

    def __call__(self, x) -> float:
        """A method to call the polynomial
        """
        evaluation = 0
        for index, coefficient in enumerate(self.coefficients):
            evaluation += coefficient * x ** index
        return evaluation

    def plotter(self) -> None:
        """Sample plotter. Don't use this."""
        x = np.linspace(0.5 * min(self.x_values), 1.25 * max(self.x_values),
                        100, endpoint=True)
        f = self(x)
        plt.plot(x, f)

        plt.show()
