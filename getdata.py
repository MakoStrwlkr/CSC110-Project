import csv
from typing import Any
import co2_read

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

def getdata() -> Any:
    """use this function to fetch all the data needed.
    """
    res = list()

    #get CO2 data
    res.extend(co2_read.read_csv("annual-co-emissions-by-region/annual-co-emissions-by-region.csv", "Brazil"))

