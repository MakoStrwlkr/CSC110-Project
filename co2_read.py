# from mpl_toolkits.basemap import Basemap, cm
# import matplotlib.pyplot as plt
import csv
from typing import Any
from getdata import Climate

def read_csv(filepath: str, country: str) -> Any:
    """read CO2 data from given csv file. 
    """
    res = list()
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            if row["Entity"] == country:
                climate = Climate(name=country, year=row["Year"], value=row["Annual CO2 emissions"])
                res.append(climate)

    return res

if __name__ == "__main__":
    print(read_csv("annual-co-emissions-by-region/annual-co-emissions-by-region.csv", "Brazil")[1])