"""
This file is for the final project in CSC110 at the University of Toronto St. George
campus. For more information, please consult the course syllabus.

This file is Copyright (c) 2020 by Ching Chang, Letian Cheng, Arkaprava Choudhury, and Hanrui Fan.
"""

from get_data.get_data_nohdf import read_data_from_csv
from polynomial_regression import PolynomialRegression
from output.output import interactive_model, prompt_independent, prompt_dependent

climates = read_data_from_csv('data/dataset.csv')

data = {
    'Annual CO2 emissions of Brazil': {},
    'Estimated Natural Forest Cover': {},
    'Amazon Precipitation': {}
}

for climate in climates:
    data[climate.name][climate.year] = climate.value

independent = prompt_independent()
dependent = prompt_dependent(independent)

x_values = [data[independent][year] for year in data[independent] if year in data[dependent]]
y_values = [data[dependent][year] for year in data[dependent] if year in data[independent]]
years = [year for year in data[independent] if year in data[dependent]]

poly = PolynomialRegression({independent: x_values}, {dependent: y_values}, 10)

interactive_model(poly, years)
