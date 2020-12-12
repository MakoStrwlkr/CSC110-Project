from getdata import Climate
import numpy as np
from pyhdf.SD import SD, SDC
import csv
import os
from typing import Any, List, Tuple, Dict


class Precipitation(Climate):
    """A class that save all of the precipitation data by longitude and latitude

    Private Attributes:
        - area: a list of list contain latitude and longitude corresponding x and y
    """
    data: Dict[Tuple[int, int], List[float]]

    def __init__(self, name: str, year: int) -> None:
        """Initialize a new precipitation dataset

        The dataset starts with no data
        """
        Climate.__init__(self, name, year)
        self.data = {}

    def add_area(self, location: Tuple[float, float], filepath: str) -> None:
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

        if (x, y) in self.data:
            print(str((latitude, longitude)) + ' already exists.')
        else:
            self.read_file(filepath, x, y)
            print(str((latitude, longitude)) + ' added.')

    def read_file(self, filepath: str, x: int, y: int) -> None:
        path_dir = os.listdir(filepath)
        cur = 0

        for s in path_dir:
            new_dir = os.path.join(filepath, s)
            if os.path.isfile(new_dir)\
                    and os.path.splitext(new_dir)[1].lower() == ".hdf"\
                    and str(self.year) in s\
                    and cur == 0:
                self.data[(x, y)] = (np.transpose(SD(new_dir, SDC.READ).select('precipitation')[:])[x][y])
                cur = 1
            elif os.path.isfile(new_dir) \
                    and os.path.splitext(new_dir)[1].lower() == ".hdf" \
                    and str(self.year) in s\
                    and cur == 1:
                temp_data = []
                temp_data.extend(np.transpose(SD(new_dir, SDC.READ).select('precipitation')[:]))
                self.data[(x, y)].append(temp_data[x][y])

    def get_value(self) -> None:
        num_point = len(self.data)
        num_month = 12
        sum_so_far = 0
        for point in self.data:
            sum_so_far += self.data[point]

        self.value = sum_so_far / num_month / num_point


if __name__ == '__main__':
    temp1 = Precipitation('AmazonPrecipitation20040201', 2004)
    temp1.add_area((50, -180), '3B43_rainfall')
    temp1.add_area((50, -179.75), '3B43_rainfall')
