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

from typing import List, Any
import numpy as np


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


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=True)

    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['numpy', 'typing'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
