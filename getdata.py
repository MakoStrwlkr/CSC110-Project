import csv
from typing import Any

class Climate:
    """A class that save all of the precipitation data by longitude and latitude

        Public Attributes:
            - name: the name of the data(e.g: CO2)
            - years: the year of the data
            - value: the value of the data

        Representation Invariants:
            - self.name != ''
            - 1986 <= self.years <= 2018
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

def co2_read_csv(filepath: str, country: str) -> Any:
    """read CO2 data from given csv file. 
    """
    res = list()
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            if row["Entity"] == country:
                climate = Climate(name="Annual CO2 emissions of " + country,\
                     year=row["Year"], value=row["Annual CO2 emissions"])
                res.append(climate)

    return res

def deforestation_read_csv(filepath: str, row_name: str) -> Any:
    """read deforestation from given csv file
    """
    res = list()
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            climate = Climate(name=row_name, year=row["Period"], value=row["Estimated Natural Forest Cover"])
            res.append(climate)

    return res


def getdata() -> Any:
    """use this function to fetch all the data needed.
    """
    res = list()

    #get CO2 data
    res.extend(co2_read_csv("annual-co-emissions-by-region/annual-co-emissions-by-region.csv", "Brazil"))

