#test data

import numpy as np
import pandas as pd

from lib import composition as hm


print(hm)

#variables
sampling_rate=20
high_cut_off = 6
low_cut_off = 0.1
filter_order = 5

# read a sample data set of 100 points 
pathIn = 'lib/data/sample.csv'
sample_data = pd.read_csv(pathIn)

# convert to arrays
x_data = sample_data["acceleration_x"].to_numpy()
y_data = sample_data["acceleration_y"].to_numpy()
z_data = sample_data["acceleration_z"].to_numpy()


impact = hm.impact_on_bone_from_acceleration(x_data, y_data, z_data, sampling_rate, low_cut_off, high_cut_off, filter_order)

print(impact)




