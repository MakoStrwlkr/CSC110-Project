# from mpl_toolkits.basemap import Basemap, cm
# import matplotlib.pyplot as plt
import numpy as np
from pyhdf.SD import SD, SDC
import csv
import os
from typing import Any, List, Tuple
import datetime
from getdata import Climate


class Co2emission(Climate):
    """A class that save all of the precipitation data
    """

    def __init__(self, name: str, date: datetime.date, stored_data: Any) -> None:
        """Initialize a new co2 dataset

        The dataset starts with no data
        """
        Climate.__init__(self, name, date, stored_data)
        self._area = []

    def add_area(self, location: str) -> None:
        """Add the given location(str) as a country

        Representation Invariants:
            - location in self.stored_data

        """
        if location.lower() not in self._area and location.lower() in self.stored_data:
            self.area_data.append(self.stored_data[location.lower()])
            self._area.append(location.lower())
            print(location + ' added.')
            self.renew_property()
        else:
            print(location + ' already exists.')

def read_csv(filepath: str, file_class: str) -> Any:
    """Read CSV File by file_class with update compatibility if there are multiple file_class

    """
    if file_class == 'co2emission':
        database = {}
        co2_year = []
        with open(filepath) as file:
            reader = csv.reader(file)
            next(reader)
            # Form self.data
            for row in reader:
                if row[1] != '' and row[3] != '0' and row[2] not in database:
                    database[row[2]] = {row[0].lower(): float(row[3])}
                elif row[1] != '' and row[3] != '0':
                    database[row[2]][row[0].lower()] = float(row[3])
            # Form Co2emission
            for year in database:
                co2_year.append(Co2emission('Co2emission' + year,
                                            datetime.date(int(year), 12, 31),
                                            database[year]))
        return co2_year


def read_csv_file(filepath: str, file_class: str) -> list:
    """Read file under folder of filepath

    """
    path_dir = os.listdir(filepath)
    data_base = []

    for s in path_dir:
        new_dir = os.path.join(filepath, s)
        if os.path.isfile(new_dir)\
                and os.path.splitext(new_dir)[1].lower() == ".csv":
            data_base = read_csv(new_dir, file_class)

    return data_base

if __name__ == '__main__':
    dataset2 = read_csv_file('annual-co-emissions-by-region', 'co2emission')
    temp2 = dataset2[55]  # 2004-12-31 co2 emission
    temp2.add_area('Brazil')
    temp2.add_area('Colombia')
    temp2.add_area('NoCountryHere')
    temp2.add_area('Brazil')
    print(temp2._area[0])
    print(dataset2[55].area_data[0])