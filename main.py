from getdata_nohdf import read_data_from_csv
from polyreg_v2_matrices import PolynomialRegression
from output import interactive_model, prompt_independent, prompt_dependent

climates = read_data_from_csv('dataset.csv')


data = {
    'Annual CO2 emissions of Brazil': {},
    'Estimated Natural Forest Cover': {},
    'Amazon Precipitation': {}
}

forest_area = {}
precipitation = {}

for climate in climates:
    data[climate.name][climate.year] = climate.value

independent = prompt_independent()
dependent = prompt_dependent(independent)

x_values = [data[independent][year] for year in data[independent] if year in data[dependent]]
y_values = [data[dependent][year] for year in data[dependent] if year in data[independent]]
years = [year for year in data[independent] if year in data[dependent]]

poly = PolynomialRegression({independent: x_values}, {dependent: y_values}, 10)
poly.plotter()

interactive_model(poly, years)
