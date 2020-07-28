#test data

import lib.accelsim as sim
import lib.composition as hm
"""
USES:

def physical_activity_counts_from_acceleration(time, accel_x, accel_y, accel_z, time_window=60, rectify="full", integration_method="simpson", plot=False, fig_size=(10,5)):

"""

sampling_rate=100
duration_in_seconds = 600 # 10 minutes

# returns time, x, y, z => data[0], data[1], data[2], dta[3]
data = sim.create3AxisAccelData(sampling_rate, duration_in_seconds, add_noise=True)


# use defaults
physical_activity = hm.physical_activity_counts_from_acceleration(data[0], data[1], data[2], data[3])

print(physical_activity)




