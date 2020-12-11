""" CSC110 Course Project - Data Analysis

Polynomial regression implemented using matrices.

In contrast to v1, where the program used lists to represent polynomials,
this program uses a different approach to finding the polynomial regression model.

A simpler approach is used here, relying more on numpy's matrix feature,
which I plan to rewrite using my own functions, similar to the transpose functions
used in Assignment 4.

The sole limitation of this model lies in calculating the errors; the complexity of
the calculations involved increases with this approach as the number of calculations changes.

Overall, I'd say this method is better, But I leave it up to you to decide, when we have
our final meeting, and reflect upon how well both these models work / don't work.

"""

import numpy as np
# import math
# import matplotlib as plt
from typing import List, Tuple


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
    multiple = [[0 for _ in range(len(matrix_2[0]))] for _ in range(len(matrix_1))]

    for i in range(len(matrix_1[0]) + 1):
        for j in range(len(matrix_2) + 1):
            for k in range(len(matrix_2)):
                multiple[i][j] += matrix_1[i][k] * matrix_2[k][j]
    return multiple


def matrix_multiplication_numpy(matrix_1: List[List[float]], matrix_2: List[List[float]]) \
        -> List[List[float]]:
    """Return the matrix matrix_1 * matrix_2, where * represents matrix multiplication.

    Preconditions:
      - len(matrix_1[0]) == len(matrix_2)

    >>> mat_1 = [[1], [1]]
    >>> mat_2 = [[1, 1]]
    >>> matrix_multiplication_small(mat_1, mat_2)
    [[1, 1], [1, 1]]
    """
    m1 = np.array(matrix_1)
    m2 = np.array(matrix_2)
    return list(np.multiply(m1, m2))


def matrix_inverse_numpy(matrix: List[List[float]]) -> List[List[float]]:
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
    matrix_array = np.array(matrix)
    matrix_inv_numpy = np.linalg.inv(matrix_array)
    return list(matrix_inv_numpy)


def find_coefficients(x_data: List[float], y_data: List[float], degree: int) -> List[float]:
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
    multiply = matrix_multiplication_small(x_matrix, x_transpose)

    inv_multiply = matrix_inverse_numpy(multiply)
    inv_multiply_with_transpose = matrix_multiplication_small(inv_multiply, x_transpose)

    beta = matrix_multiplication_small(inv_multiply_with_transpose, y_matrix)

    return [beta[i][0] for i in range(len(beta))]


class Matrix:
    """An abstract data class for a nested list representation of a matrix.
    All subclasses support matrix transpose, matrix inverse, matrix multiplication.

    Public Instance Attributes:
      - matrix: nested list representation of a matrix.
    """

    ...


class XMatrix(Matrix):
    """Matrix for storing the powers of x.

    Public instance attributes:
      - matrix: nested list representation of a matrix
    """
    matrix: List[List[float]]

    def __init__(self, points: List[float], degree: int) -> None:
        """Returns a nested list representation of a matrix consisting of the basis
        elements of the polynomial of degree n, evaluated at each of the points.

        In other words, each row consists of 1, x, x^2, ..., x^n, where n is the degree,
        and x is a value in points.

        Preconditions:
          - degree < len(points)
        """
        self.matrix = make_matrix(points, degree)


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
