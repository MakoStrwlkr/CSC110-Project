""" CSC110 Course Project - Read Data and Store with Defined Type

"""
import csv
from typing import Any, List


class Climate:
    """A class that save all of the precipitation data by longitude and latitude

        Public Attributes:
            - name: the name of the data(e.g: CO2)
            - year: the year of the data
            - value: the value of the data

        Representation Invariants:
            - self.name != ''
    """
    name: str
    year: int
    value: float

    def __init__(self, name: str, year: int, value: float) -> None:
        self.name = name
        self.year = year
        self.value = value

    def __str__(self) -> str:
        return str([self.name, self.year, self.value])

    def __lt__(self, other: Any) -> bool:
        return self.year < other.year


def read_data_from_csv(filename: str) -> List[Climate]:
    """read the data from saved csv
    """
    res = list()
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            climate = Climate(name=row["name"], year=int(row["year"]), value=float(row["value"]))
            res.append(climate)

    return res


if __name__ == '__main__':
    # sample usage
    dataset = read_data_from_csv("dataset.csv")

    import doctest

    doctest.testmod(verbose=True)

    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['numpy', 'matplotlib.pyplot', 'typing', 'math',
                          "pyhdf.SD", "pyhdf.SDC", "csv", "os"],
        'allowed-io': ["read_data_from_csv", "save_data_as_csv",
                       "deforestation_read_csv", "co2_read_csv"],  # the names (strs) of functions that call print/open/input
        'max-line-length': 150,
        'disable': ['R1705', 'C0200'],
        'max-nested-blocks': 5
    })
