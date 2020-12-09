# from mpl_toolkits.basemap import Basemap, cm
# import matplotlib.pyplot as plt
import numpy as np
from pyhdf.SD import SD, SDC
import os
from typing import Any, List, Dict, Tuple
import datetime


class Precipitation:
    """A class that save all of the precipitation data by longitude and latitude

    Public Attributes:
        - name: the name of this precipitation dataset
        - area: a list of list contain latitude and longitude corresponding x and y
        - date: a datetime object record date of
        - precip: a list of float corresponding to area added
        - average: a float represent the average of precipitation in the area
        - max: a float represent the max of precipitation in the area
        - min: a float represent the min of precipitation in the area
        - dataset: the precipitation data for different years

    Representation Invariants:
        - self.name != ''
    """
    name: str
    date: datetime.date
    area: List[Tuple[int, int]]
    precip: List[float]
    average: float
    max: float
    min: float
    dataset: Any

    def __init__(self, name: str, date: datetime.date, data: Any) -> None:
        """Initialize a new precipitation dataset

        The dataset starts with no data
        """
        self.name = name
        self.date = date
        self.area = []
        self.precip = []
        self.average = 0.0
        self.max = 0.0
        self.min = 0.0
        self.dataset = data

    def add_area(self, latitude: float, longitude: float) -> None:
        """Add the given location responding grid by 0.25°*0.25° in scope of 50°S to 50°N and 180°W – 180°E
        S is negative and W correspond to negative value.

        Representation Invariants:
            - -50 <= self.latitude <= 50
            - -180 <= self.longitude <= 180
        """
        if latitude == -50:
            latitude += 0.01
        if longitude == 180:
            longitude = -longitude

        x = int((-(latitude - 50)) // 0.25)
        y = int((longitude + 180) // 0.25)
        location = (x, y)

        if location in self.area:
            print(str((latitude, longitude)) + ' already exists.')
        else:
            self.area.append(location)
            print(str((latitude, longitude)) + ' added.')

            temp_precip = self.dataset[x][y]
            self.precip.append(temp_precip)

            self.max, self.min, self.average = self.property()

    def property(self) -> Tuple[float, float, float]:
        """Return max, min and average of precipitation

        """
        import statistics
        return (max(self.precip), min(self.precip), statistics.mean(self.precip))


def read_each_file(filepath: str, file_class: str, item_name: str) -> list:
    """Read file under folder of filepath

    """
    path_dir = os.listdir(filepath)
    data_set = []

    for s in path_dir:
        new_dir = os.path.join(filepath, s)
        if os.path.isfile(new_dir):
            if os.path.splitext(new_dir)[1].lower() == ".hdf" and file_class.lower() == 'precipitation':
                date = os.path.splitext(new_dir)[0].split('.')[1]
                data_set.append(Precipitation(item_name + 'Precipitation' + date,
                                              datetime.date(int(date[0:4]), int(date[4:6]), int(date[6:8])),
                                              read_hdf(new_dir, file_class.lower())))

    return data_set


def read_hdf(filepath: str, file_class: str) -> Any:
    """Read HDF File

    """
    if file_class == 'precipitation':
        precipitation_dataset = np.transpose(SD(filepath, SDC.READ).select('precipitation')[:])
        return precipitation_dataset


if __name__ == '__main__':
    data = np.transpose(SD('3B43_rainfall/3B43.20040201.7A.HDF', SDC.READ).select('precipitation')[:])
    temp = Precipitation('AmazonPrecipitation20040201', datetime.date(2004, 2, 1), data)
    temp.add_area(50, -180)
    temp.add_area(50, -179.75)
    temp.add_area(50, -179.49)
    temp.add_area(50, -179.76)
    dataset = read_each_file('3B43_rainfall', 'precipitation', 'Amazon')
