"""
This file is for the final project in CSC110 at the University of Toronto St. George
campus. For more information, please consult the course syllabus.

This file is Copyright (c) 2020 by Ching Chang, Letian Cheng, Arkaprava Choudhury, and Hanrui Fan.
"""

import csv
import os
from typing import Any, List, Tuple

import numpy as np
from pyhdf.SD import SD, SDC


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


def precipitation_read_hdf(filepath: str,
                           leftup: Tuple[float, float],
                           rightbottom: Tuple[float, float]) -> List[Climate]:
    """Read precipitation data from given hdf file. And select a rectangle
    area by given leftup and rightbottom.

    Add the given location(Tuple) responding grid by 0.25°*0.25° in scope
    of responding grid by 0.25°*0.25° in scope of 50°S to 50°N and 180°W
    – 180°E. S is negative and W correspond to negative value.

    Representation Invariants:
        - -50 <= leftup[0] <= 50
        - -50 <= rightbottom[0] <= 50
        - -180 <= leftup[1] <= 180
        - -180 <= rightbottom[1] <= 180
        - leftup[0] < rightbottom[0]
        - leftup[1] > rightbottom[1]
    """
    res = list()

    if leftup[0] == -50:
        leftup = (leftup[0] + 0.01, leftup[1])
    if leftup[1] == 180:
        leftup = (leftup[0], -leftup[1])
    if rightbottom[0] == -50:
        rightbottom = (rightbottom[0] + 0.01, rightbottom[1])
    if rightbottom[1] == 180:
        rightbottom = (rightbottom[0], -rightbottom[1])

    p1 = (int((-(leftup[0] - 50)) // 0.25), int((leftup[1] + 180) // 0.25))
    p2 = (int((-(rightbottom[0] - 50)) // 0.25), int((rightbottom[1] + 180) // 0.25))

    path_dir = os.listdir(filepath)
    stored_data = {}
    for s in path_dir:
        new_dir = os.path.join(filepath, s)
        year = os.path.splitext(new_dir)[0].split('.')[1][0:4]
        if os.path.splitext(new_dir)[1].lower() == ".hdf" and year not in stored_data:
            stored_data[year] = 0.0
            stored_data[year] += process_hdf_data(new_dir, p1, p2)
        elif os.path.splitext(new_dir)[1].lower() == ".hdf":
            process_hdf_data(new_dir, p1, p2)
            stored_data[year] += process_hdf_data(new_dir, p1, p2)

    for year in stored_data:
        res.append(Climate('Amazon Precipitation', int(year),
                           stored_data[year] / 12 / (p2[0] - p1[0] + 1) / (p2[1] - p1[1] + 1)))

    res.sort()
    return res


def process_hdf_data(filepath: str, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """A helper function of precipitation_read_hdf to calculate the sum of all
    precipitation of provided p1(position1) and p2(position2).
    """
    temp = np.transpose(SD(filepath, SDC.READ).select('precipitation')[:])
    sum_so_far = 0.0

    for i in range(p1[0], p2[0] + 1):
        for j in range(p1[1], p2[1] + 1):
            sum_so_far += temp[i][j]

    return sum_so_far


def co2_read_csv(filepath: str, country: str) -> Any:
    """read CO2 data from given csv file.
    In this project, you should always choose country as Brazil.
    """
    res = list()
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            if row["Entity"] == country:
                climate = Climate(name="Annual CO2 emissions of " + country,
                                  year=int(row["Year"]), value=float(row["Annual CO2 emissions"]))
                res.append(climate)

    res.sort()
    return res


def deforestation_read_csv(filepath: str, row_name: str) -> List[Climate]:
    """read deforestation from given csv file.
    You can choose the row name from the following:
        - Period
        - Estimated Natural Forest Cover
        - Deforestation (INPE)
        - Natural forest cover change
        - Forest cover as % of pre-1970 cover
        - Total forest loss since 1970
    """
    res = list()
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            climate = Climate(name=row_name, year=int(row["Period"]),
                              value=float(row[row_name].replace(',', '')))
            res.append(climate)

    res.sort()
    return res


def getdata() -> List:
    """Fetch all the data needed.
    res is a list contains Climate class.
    """
    res = list()

    # get CO2 data
    res.extend(co2_read_csv('data/annual-co-emissions-by-region.csv',
                            'Brazil'))

    # get deforestation data
    res.extend(deforestation_read_csv('data/deforestation.csv', 'Estimated Natural Forest Cover'))

    # get precipitation data
    res.extend(precipitation_read_hdf('data/3B43_rainfall', (0.553222, -65.162917),
                                      (-4.070444, -52.109639)))

    return res


def save_data_as_csv(data: list, filename: str) -> None:
    """Save the data from getdata() as csv.
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ["name", "year", "value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')

        writer.writeheader()
        for i in data:
            writer.writerow({"name": i.name, "year": i.year, "value": i.value})


def read_data_from_csv(filename: str) -> List[Climate]:
    """Read the data from saved csv
    """
    res = list()
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            climate = Climate(name=row["name"], year=int(row["year"]), value=float(row["value"]))
            res.append(climate)

    return res


if __name__ == '__main__':
    save_data_as_csv(getdata(), "data/dataset.csv")
    # read_data_from_csv("data/dataset.csv")

    import doctest

    doctest.testmod(verbose=True)

    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['numpy', 'matplotlib.pyplot', 'typing', 'math',
                          "pyhdf.SD", "pyhdf.SDC", "csv", "os"],
        'allowed-io': ["read_data_from_csv", "save_data_as_csv",
                       "deforestation_read_csv", "co2_read_csv"],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
