""" CSC110 Course Project - Read Data and Store with Defined Type

"""
import numpy as np
import csv
import os
from typing import List, Tuple, Any


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
    dataset = read_data_from_csv("dataset.csv")
