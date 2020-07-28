#test data
import accelsim as sim
import plot as plt
import gait as gt
import signalproc as sp
import peakdetection as pk
import gait as g
import integration as integrate
import plot as p


"""

General process

build data set (or get data set)
filter it 
process with fft
do some computation

"""

sampling_rate=100

# returns time, x, y, z => data[0], data[1], data[2], dta[3]
data = sim.create3AxisAccelData(sampling_rate, 600, add_noise=True)
# 600 seconds = 10 minutes

p.plot_signal(data[0], [{'data': data[1], 'label': 'Medio-lateral (ML) - side to side', 'line_width': 0.5},
                           {'data': data[2], 'label': 'Vertical (VT) - up down', 'line_width': 0.5},
                           {'data': data[3], 'label': 'Antero-posterior (AP) - forwards backwards', 'line_width': 0.5}],
                    subplots=True, fig_size=(10,7))
#b, a = sp.build_filter(6, sampling_rate, 'low', filter_order=5)


# Filter signals
#x_f = sp.filter_signal(b, a, data[1], "filtfilt")  # ML medio-lateral
#y_f = sp.filter_signal(b, a, data[2], "filtfilt")  # VT vertical
#z_f = sp.filter_signal(b, a, data[3], "filtfilt")  # AP antero-posterior



x_i = integrate.integrate_signal(data[1], data[0], plot=False)






# peak detection in vertical (up and down)
#peak_times, peak_values = pk.find_peaks(data[0], y_f, peak_type='both', min_val=0.8, min_dist=10, plot=True)


# gait statistics


# all off peak_times -> the times that peaks occur not the values of the peaks
#step_count = g.step_count(peak_times)
#cadence = g.cadence(time, peak_times)
#step_time, step_time_sd, step_time_cov = g.step_time(peak_times)


#print(data)

#plot signal

#plt.plot_signal(data[0], [{'data': data[1], 'label': 'Medio-lateral (ML) - side to side', 'line_width': 0.5},
#                           {'data': data[2], 'label': 'Vertical (VT) - up down', 'line_width': 0.5},
#                           {'data': data[3], 'label': 'Antero-posterior (AP) - forwards backwards', 'line_width': 0.5}],
#                    subplots=True, fig_size=(10,7))

# plot fiter responses
# plt.plot_filter_response((0.01,6),20,"bandpass", filter_order=5)
# plt.plot_filter_response(0, 20, "low", filter_order=5)