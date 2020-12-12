# from mpl_toolkits.basemap import Basemap, cm
# import matplotlib.pyplot as plt
import numpy as np
from pyhdf.SD import SD, SDC
import csv
import os
from typing import Any, List, Tuple
import datetime


class Climate:
    """A class that save all of the precipitation data by longitude and latitude

        Public Attributes:
            - name: the name of this Climate dataset
            - date: a datetime object record date
            - stored_data: the precipitation/co2emission data for given time
            - data: a list of float corresponding to area added
            - average: a float represent the average of precipitation in the area
            - max: a float represent the max of precipitation/co2 in the area
            - min: a float represent the min of precipitation/co2 in the area

        Representation Invariants:
            - self.name != ''
    """
    name: str
    date: datetime.date
    stored_data: Any
    area_data: List[float]
    average: float
    max: float
    min: float

    def __init__(self, name: str, date: datetime.date, stored_data: Any) -> None:
        """Initialize a new precipitation dataset

        The dataset starts with no data
        """
        self.name = name
        self.date = date
        self.stored_data = stored_data
        self.area_data = []
        self.average = 0.0
        self.max = 0.0
        self.min = 0.0

    def add_area(self, location: Any) -> None:
        """Add the area that needed to be examined.

        """
        raise NotImplementedError

    def renew_property(self) -> None:
        """Renew max, min and average of data

        """
        import statistics
        self.max = max(self.area_data)
        self.min = min(self.area_data)
        self.average = statistics.mean(self.area_data)


class Co2emission(Climate):
    """A class that save all of the precipitation data by longitude and latitude

        Private Attributes:
            - area: a list of string contains country
    """
    _area: List[str]

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


class Precipitation(Climate):
    """A class that save all of the precipitation data by longitude and latitude

    Private Attributes:
        - area: a list of list contain latitude and longitude corresponding x and y
    """
    _area: List[Tuple[int, int]]

    def __init__(self, name: str, date: datetime.date, stored_data: Any) -> None:
        """Initialize a new precipitation dataset

        The dataset starts with no data
        """
        Climate.__init__(self, name, date, stored_data)
        self._area = []

    def add_area(self, location: Tuple[float, float]) -> None:
        """Add the given location(Tuple) responding grid by 0.25°*0.25° in scope of 50°S to 50°N and 180°W – 180°E
        S is negative and W correspond to negative value.

        Representation Invariants:
            - -50 <= location[0] <= 50
            - -180 <= location[1] <= 180
        """
        latitude = location[0]
        longitude = location[1]
        if latitude == -50:
            latitude += 0.01
        if longitude == 180:
            longitude = -longitude

        x = int((-(latitude - 50)) // 0.25)
        y = int((longitude + 180) // 0.25)
        location = (x, y)

        if location in self._area:
            print(str((latitude, longitude)) + ' already exists.')
        else:
            self._area.append(location)
            print(str((latitude, longitude)) + ' added.')

            temp_precip = self.stored_data[x][y]
            self.area_data.append(temp_precip)
            self.renew_property()


def read_each_file(filepath: str, file_class: str) -> list:
    """Read file under folder of filepath

    """
    path_dir = os.listdir(filepath)
    data_base = []

    for s in path_dir:
        new_dir = os.path.join(filepath, s)
        if os.path.isfile(new_dir)\
                and os.path.splitext(new_dir)[1].lower() == ".hdf"\
                and file_class.lower() == 'precipitation':
            date = os.path.splitext(new_dir)[0].split('.')[1]
            data_base.append(Precipitation('Precipitation' + date,
                                          datetime.date(int(date[0:4]), int(date[4:6]), int(date[6:8])),
                                          read_hdf(new_dir, file_class.lower())))
        elif os.path.isfile(new_dir)\
                and os.path.splitext(new_dir)[1].lower() == ".csv":
            data_base = read_csv(new_dir, file_class)

    return data_base


def read_hdf(filepath: str, file_class: str) -> Any:
    """Read HDF File by file_class with update compatibility if there are multiple file_class

    """
    if file_class == 'precipitation':
        precipitation_dataset = np.transpose(SD(filepath, SDC.READ).select('precipitation')[:])
        return precipitation_dataset


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



if __name__ == '__main__':
    data1 = np.transpose(SD('3B43_rainfall/3B43.20040201.7A.HDF', SDC.READ).select('precipitation')[:])
    temp1 = Precipitation('AmazonPrecipitation20040201', datetime.date(2004, 2, 1), data1)
    temp1.add_area((50, -180))
    temp1.add_area((50, -179.75))
    temp1.add_area((50, -179.49))
    temp1.add_area((50, -179.76))
    dataset1 = read_each_file('3B43_rainfall', 'precipitation')
    dataset2 = read_each_file('annual-co-emissions-by-region', 'co2emission')
    temp2 = dataset2[55]  # 2004-12-31 co2 emission
    temp2.add_area('Brazil')
    temp2.add_area('Colombia')
    temp2.add_area('NoCountryHere')
    temp2.add_area('Brazil')
