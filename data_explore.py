import numpy as np
import pandas as pd
import lib.signalproc as sp
import matplotlib.pyplot as plt
import lib.plot as p
import lib.peakdetection as pk

from statistics import mean 


# read a sample data set 

# filename structure subject1_ideal = subject 1 with ideal placement of sensors

# sensor of interest is the BACK

# 120 columns in the dataset

# first column = the second of interest 0 = first second, 1 = second second etc

# second column = the time in micro seconds. first sample is at 20000 or 20 ms or 0.02 seconds uS to ms /1000. ms to seconds /1000

# last column is the activity

# accel X, accel y, accel z, gyro X:Z, mag X:Z, quarterniion1, q2, q3, q4 

# back sensor is the 3rd group of 13 so starts at 1,2 + 26 = 29->41

def quaternion_to_euler_angle_vectorized1(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z 

def plot_signals(x, y, z):
	fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(100,9))

	ax[0].set_title('x signal')
	ax[0].plot(x, linewidth=0.3, color='k')
	#ax[0].plot(time, y_f, linewidth=0.8, color='r')

	ax[1].set_title("y signal")
	ax[1].plot(y, linewidth=0.3, color='b')

	ax[2].set_title("z signal")
	ax[2].plot(z, linewidth=0.3, color='g')

	fig.subplots_adjust(hspace=.5)

	plt.show()


def plot_signals_and_filters(x, x_f, y, y_f, z, z_f):
	fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(100,9))

	ax[0].set_title('x signal')
	ax[0].plot(x, linewidth=0.3, color='k')
	ax[0].plot(time, x_f, linewidth=0.8, color='r')

	ax[1].set_title("y signal")
	ax[1].plot(y, linewidth=0.3, color='b')
	ax[0].plot( y_f, linewidth=0.8, color='r')

	ax[2].set_title("z signal")
	ax[2].plot(z, linewidth=0.3, color='g')
	ax[0].plot(z_f, linewidth=0.8, color='r')

	fig.subplots_adjust(hspace=.5)

	plt.show()

def plot_single_signal_and_filtered(s, s_f):
	plt.figure(figsize=(100,10)) 
	plt.plot(s, 'b') # plotting t, a separately 
	plt.plot(s_f, 'r') # plotting t, b separately 
	plt.show()

def get_data(col_names):
	pathIn = 'lib/data/subject1_ideal.csv'
	sample_data = pd.read_csv(pathIn, names = col_names )
	return sample_data


def create_columns_for_data():
	sensors = ['RLA', 'RUA', 'BACK', 'LUA', 'LLA', 'RC', 'RT', 'LT', 'LC']

	measurement= ['accel_x', 'accel_y', 'accel_z', 'gryo_x', 'gryo_y','gryo_z', 'mag_x', 'mag_y', 'mag_z', 'quat_w', 'quat_x', 'quat_y', 'quat_z']

	column_names = []

	column_names.append('second')
	column_names.append('micoseconds')

	for sensor in sensors:
		for m in measurement:
			col_name = m + "_" + sensor
			column_names.append(col_name)

	column_names.append("activity_label")
	return column_names


def reduce_data_by_sensor_and_modality_and_activity(data, sensor, modality, activity):
	if(modality == "accel"):
		data = data.filter(["second", "micoseconds", modality + "_x_" + sensor, modality + "_y_" + sensor, modality + "_z_" + sensor, 'activity_label'])
		new = data[data.activity_label == activity]
		return new
	elif(modality == "quat"):
		data = data.filter(["second", "micoseconds", modality + "_w_" + sensor, modality + "_x_" + sensor, modality + "_y_" + sensor, modality + "_z_" + sensor, 'activity_label'])
		new = data[data.activity_label == activity]
		return new



def convert_3d_data_to_arrays(df, sensor, modality):
	if(modality == "accel"):
		x = df[modality +"_x_" + sensor].to_numpy()
		y = df[modality +"_y_" + sensor].to_numpy()
		z = df[modality +"_z_" + sensor].to_numpy()
		return x,y,z
	elif(modality == "quat"):
		w = df[modality +"_w_" + sensor].to_numpy()
		x = df[modality +"_x_" + sensor].to_numpy()
		y = df[modality +"_y_" + sensor].to_numpy()
		z = df[modality +"_z_" + sensor].to_numpy()
		return w, x, y, z


def peak_pairs_from_peaks(peaks):
	peak_pairs = []
	n=2
	for i,x in enumerate(peaks):
	    if i % n == 0:
	    	pair = (peaks[i], peaks[i+1])
	    	peak_pairs.append(pair)
	return peak_pairs

def pair_differences(pairs):
	pair_differences = []

	for pair in pairs:
		#diff = abs(pair[0]-pair[1])
		diff = pair[0] - pair[1]
		pair_differences.append(diff)
	return pair_differences


# composition function to compute the imbalance of the pelvice (or whatever limb the sensor is attached too.)
def computes_axis_difference(time, axis, low_cut_off_frequency, high_cut_off_frequency, sampling_rate, filter_order):

	b, a = sp.build_filter((low_cut_off_frequency, high_cut_off_frequency), sampling_rate, 'bandpass', filter_order)
	axis_f = sp.filter_signal(b,a, axis, "lfilter")
	#plot_single_signal_and_filtered(roll, roll_f)

	rectified_signal = sp.rectify_signal(axis_f, rectifier_type="full", plot=True, show_grid=True, fig_size=(10, 5))

	peak_times, peak_values = pk.find_peaks(time, rectified_signal, peak_type='peak', min_val=0.3, min_dist=5, plot=True)

	# difference between each pair of values i.e 0,1   2,3 etc

	if((len(peak_values) % 2) == 1):
		peak_values = peak_values[:len(peak_values)-1]

	peak_pairs = peak_pairs_from_peaks(peak_values)

	# what does the direction mean?
	difference = pair_differences(peak_pairs)

	return difference
# q1 = w,
# q2 = x
# q3 = y
# q4 = z
# variables


sampling_rate = 50 # 50 Hz  0.02 seconds
low_cut_off_frequency = 0.3 # works alot better for smoothing the signal
high_cut_off_frequency = 5 # check fft for this value
filter_order = 5

# prepare data

# 1 minute of walking data

cols = create_columns_for_data()
data = get_data(cols)
new = reduce_data_by_sensor_and_modality_and_activity(data, "RT", "quat", 1)




# create a time vector
#time = new["micoseconds"].to_numpy()


#x,y,z = convert_3d_data_to_arrays(new, "RT", "accel")

w, x, y, z = convert_3d_data_to_arrays(new, "RT", "quat")

# convert to angles

#roll,pitch,yaw = quaternion_to_euler_angle_vectorized1(w,x,y,z)

# roll pitch yaw


w_10 = w[:15*sampling_rate]
x_10 = x[:15*sampling_rate]
y_10 = y[:15*sampling_rate]
z_10 = z[:15*sampling_rate]

# 10 seconds of data
#w_10 = w[:10*sampling_rate]
#roll_10 = roll[:10*sampling_rate]
#pitch_10 = pitch[:10*sampling_rate]
#yaw_10 = yaw[:10*sampling_rate]

time_10  = np.linspace(0,  len(x_10)*1000, 15*sampling_rate )


roll,pitch,yaw = quaternion_to_euler_angle_vectorized1(w_10,x_10,y_10,z_10)

#_ = sp.fft(y_10, sampling_rate, plot=True)

#plot_signals(x,y,z)
#plot_signals(roll, pitch, yaw)


# roll and yaw give "good" results - pitch has odd number of peaks detected

difference = computes_axis_difference(time_10, roll, low_cut_off_frequency, high_cut_off_frequency, sampling_rate, filter_order  )

print(round(mean(difference),2))

#b, a = sp.build_filter(high_cut_off_frequency, sampling_rate, 'low', filter_order)
#y_f = sp.filter_signal(b, a, y_10, "filtfilt")  # ML medio-lateral




#_ = sp.fft(y_f, sampling_rate, plot=True)















