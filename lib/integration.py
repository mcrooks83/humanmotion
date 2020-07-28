from scipy.integrate import simps, trapz
import lib.signalproc as sp
import numpy as np
import matplotlib.pyplot as plt




def integrate_signal(time, data, time_scale="ms", time_window=60, rectify="full", integration_method="simpson", plot=False, fig_size=(10,5)):

	# integration of acceleration provides velocity 
	# the time_window provides the sequnce over which to integrate
	# the counts of the integrations - physical activity

	# returns the integration values for every time_window

	# make sure that the len of the data has corresponding time 
	assert len(data) == len(time), "signal and time must be the same length"

	#makes no difference!
	#x = np.asarray(data)
	#time = np.asarray(time)

	# convert time to seconds
	if time_scale == "ms":
	    time = time / 1000
	elif time_scale == "s":
	    time = time


	# rectify the signal make the negative portions positive if full or negative values 0 if half
	x = sp.rectify_signal(data, rectify)

	boundary_count = int(max(time) / time_window) + 1
	boundary_times = [i * time_window for i in range(boundary_count)]


	missing_times = np.setdiff1d(boundary_times, time)  # epoch times to interp


	x = np.append(x, np.interp(missing_times, time, x))  # interpolate x values
	time = np.append(time, missing_times)

	# sort new time and signal arrays together
	sort_idx = time.argsort()
	time = time[sort_idx]
	x = x[sort_idx]

	# get index of each epoch/boundary value for slicing
	boundary_idx = np.where(np.isin(time, boundary_times))[0]


	# integrate each epoch using Simpson's rule
	counts = np.ones(len(boundary_idx) - 1)  # preallocate array
	for i in range(len(counts)):
	    lower = boundary_idx[i]
	    upper = boundary_idx[i + 1] + 1  # upper bound should be inclusive

	    cur_signal = x[lower:upper]
	    cur_time = time[lower:upper]

	    if integration_method == "simpson":
	        counts[i] = simps(cur_signal, cur_time)
	    elif integration_method == "trapezoid":
	        counts[i] = trapz(cur_signal, cur_time)

	# plot counts
	if plot:
	    f, ax = plt.subplots(1, 1, figsize=fig_size)

	    ax.bar(boundary_times[1:], counts, width=time_window - 5)

	    plt.xticks(
	        boundary_times[1:],
	        [
	            "{} - {}".format(boundary_times[i], boundary_times[i + 1])
	            for i, x in enumerate(boundary_times[1:])
	        ],
	    )

	    for tick in ax.get_xticklabels():
	        tick.set_rotation(45)

	    plt.suptitle("Physical activity counts", size=16)
	    plt.xlabel("Time window (seconds)")
	    plt.ylabel("PA count")
	    plt.show()

	return counts
