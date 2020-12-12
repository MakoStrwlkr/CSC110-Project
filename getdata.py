import csv

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
    years: int
    value: float

    def __init__(self, name: str, years: int, value: float) -> None:
        self.name = name
        self.years = years
        self.value = value