'''
USES:

def basic_gait_analysis_from_acceletaion(time, axis_accel, sampling_rate, cut_off, filter_order):

'''

import numpy as np
import pandas as pd

from lib import composition as hm
from lib import accelsim as sim


# variables 
sampling_rate=20
cut_off = 6
filter_order= 5

duration_seconds = 30


# create some data
# 
# returns time, x, y, z => data[0], data[1], data[2], dta[3]
data = sim.create3AxisAccelData(sampling_rate, 30)


# use y (vertical) axis
gait_metrics = hm.basic_gait_analysis_from_acceletaion(data[0], data[2], sampling_rate, cut_off, filter_order)

print(gait_metrics)