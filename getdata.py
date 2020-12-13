""" CSC110 Course Project - Read Data and Store with Defined Type

"""
import numpy as np
from pyhdf.SD import SD, SDC
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


def precipitation_read_hdf(filepath: str,
                           leftup: Tuple[float, float],
                           rightbottom: Tuple[float, float]) -> List[Climate]:
    """Read precipitation data from given hdf file. And select a rectangle area by given leftup and rightbottom.
    Add the given location(Tuple) responding grid by 0.25°*0.25° in scope of responding grid by 0.25°*0.25°
    in scope of 50°S to 50°N and 180°W – 180°E. S is negative and W correspond to negative value.

        Representation Invariants:
            - -50 <= leftup[0] <= 50
            - -50 <= rightbottom[0] <= 50
            - -180 <= leftup[1] <= 180
            - -180 <= rightbottom[1] <= 180
            - leftup[0] < rightbottom[0]
            - leftup[1] > rightbottom[1]
    """
    res = list()
    lat1 = leftup[0]
    lon1 = leftup[1]
    lat2 = rightbottom[0]
    lon2 = rightbottom[1]
    if lat1 == -50:
        lat1 += 0.01
    if lon1 == 180:
        lon1 = -lon1
    if lat2 == -50:
        lat2 += 0.01
    if lon2 == 180:
        lon2 = -lon2

    x1 = int((-(lat1 - 50)) // 0.25)
    y1 = int((lon1 + 180) // 0.25)
    x2 = int((-(lat2 - 50)) // 0.25)
    y2 = int((lon2 + 180) // 0.25)

    path_dir = os.listdir(filepath)
    stored_data = {}
    for s in path_dir:
        new_dir = os.path.join(filepath, s)
        year = os.path.splitext(new_dir)[0].split('.')[1][0:4]
        if os.path.splitext(new_dir)[1].lower() == ".hdf" \
                and year not in stored_data:
            temp = np.transpose(SD(new_dir, SDC.READ).select('precipitation')[:])
            sum_so_far = 0.0
            for i in range(x1, x2 + 1):
                for j in range(y1, y2 + 1):
                    sum_so_far += temp[i][j]
            stored_data[year] = sum_so_far

        elif os.path.splitext(new_dir)[1].lower() == ".hdf":
            temp = np.transpose(SD(new_dir, SDC.READ).select('precipitation')[:])
            sum_so_far = 0.0
            for i in range(x1, x2 + 1):
                for j in range(y1, y2 + 1):
                    sum_so_far += temp[i][j]
            stored_data[year] += sum_so_far
    for year in stored_data:
        res.append(Climate('Amazon Precipitation', int(year), stored_data[year] / 12 / (x2 - x1 + 1) / (y2 - y1 + 1)))

    return res


def co2_read_csv(filepath: str, country: str) -> Any:
    """read CO2 data from given csv file.
    """
    res = list()
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            if row["Entity"] == country:
                climate = Climate(name="Annual CO2 emissions of " + country,
                                  year=int(row["Year"]), value=float(row["Annual CO2 emissions"]))
                res.append(climate)

    return res


def deforestation_read_csv(filepath: str, row_name: str) -> List[Climate]:
    """read deforestation from given csv file
    """
    res = list()
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            climate = Climate(name=row_name, year=int(row["Period"]), value=float(row[row_name].replace(',', '')))
            res.append(climate)

    return res


def getdata() -> List:
    """use this function to fetch all the data needed.
    res is a list contains Climate class.
    """
    res = list()

    # get CO2 data
    res.extend(co2_read_csv('annual-co-emissions-by-region/annual-co-emissions-by-region.csv', 'Brazil'))

    # get deforestation data
    res.extend(deforestation_read_csv('deforestation.csv', 'Estimated Natural Forest Cover'))

    # get precipitation data
    res.extend(precipitation_read_hdf('3B43_rainfall', (0.553222, -65.162917), (-4.070444, -52.109639)))

    return res


def save_data_as_csv(data: list, filename: str) -> None:
    """save the data from getdata() as csv.
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ["name", "year", "value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')

        writer.writeheader()
        for i in data:
            writer.writerow({"name": i.name, "year": i.year, "value": i.value})


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
    save_data_as_csv(getdata(), "dataset.csv")
    # read_data_from_csv("dataset.csv")
