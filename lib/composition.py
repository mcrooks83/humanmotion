#imports

import lib.gait as g
import lib.signalproc as sp
import lib.peakdetection as pk
import lib.integration as i

import numpy as np
import pandas as pd



def impact_on_bone_from_acceleration(accel_x, accel_y, accel_z, sampling_rate, low_cut_off_frequency, high_cut_off_frequency, filter_order ):

	# compute magnitdue

	magnitude_vector = sp.vector_magnitude(accel_x, accel_y, accel_z)

	# filter the magnitude vector

	b, a = sp.build_filter((low_cut_off_frequency, high_cut_off_frequency), sampling_rate, 'bandpass', filter_order)
	filtered = sp.filter_signal(b,a, magnitude_vector, "lfilter")
	# compute fft (frequency spectrun within filtered magnitude vector)

	fft_mag = sp.compute_fft_mag(filtered)

	oi = bone.compute_intensity(fft_mag, sampling_rate, high_cut_off_frequency)

	positive = bone.compute_impact(oi)

	impact = {
		"id": "impact on bone",
		"positive": positive,
		"oi": oi
	}

	return impact


def basic_gait_analysis_from_acceletaion(time, axis_accel, sampling_rate, cut_off, filter_order):

	# build filter 
	b, a = sp.build_filter(cut_off, sampling_rate, 'low', filter_order)

	# filter each axis
	f_axis = sp.filter_signal(b, a, axis_accel, "filtfilt")  # ML medio-lateral

	# get peaks
	peak_times, peak_values = pk.find_peaks(time, axis_accel, peak_type='valley', min_val=0.8, min_dist=10, plot=False)

	# step count is the number of cycles detected. 
	step_count = g.step_count(peak_times)
	cadence = g.cadence(time, peak_times)
	step_time, step_time_sd, step_time_cov = g.step_time(peak_times)

	ac, ac_lags = sp.xcorr(axis_accel, axis_accel, scale='unbiased', plot=False)

	ac_peak_times, ac_peak_values = pk.find_peaks(ac_lags, ac, peak_type='peak', min_val=0.1, min_dist=30, plot=False)

	step_reg, stride_reg = g.step_regularity(ac_peak_values)
	step_sym = g.step_symmetry(ac_peak_values)

	basic_gait_metrics = {
		"id": "Basic gait metrics",
		"numOfSteps": step_count,
		"cadence": {
			"id": "step cadence",
			"value": cadence,
			"unit": "steps / minute"
		},
		"meanStepTime": {
			"id": "mean step time",
			"value": step_time,
			"unit": "ms"
		},
		"stepVariabilitySD": {
			"id": "step variabilty standard deviation",
			"value": step_time_sd,
		},
		"stepVariabilityCoV": {
			"id": "step variabilty coefficient of variabilty",
			"value": step_time_cov
		},
		"stepRegularity": step_reg,
		"strideRegularity": stride_reg,
		"step_symmetry": step_sym
	}

	return basic_gait_metrics



def physical_activity_counts_from_acceleration(time, accel_x, accel_y, accel_z, time_scale="ms", time_window=60, rectify="full", integration_method="simpson", plot=False, fig_size=(10,5)):

	#integrate each signal

	x_i = i.integrate_signal(time, accel_x, time_scale, time_window, rectify, integration_method, plot, fig_size)
	y_i = i.integrate_signal(time, accel_y, time_scale, time_window, rectify, integration_method, plot, fig_size)
	z_i = i.integrate_signal(time, accel_z, time_scale, time_window, rectify, integration_method, plot, fig_size)


	magnitudes = sp.vector_magnitude(x_i, y_i, z_i)

	physical_activity_data = {
		"id": "physical activity data",
		"xAxisCounts": x_i,
		"yAxisCounts": y_i,
		"zAxisCounts": z_i,
		"magnitudeVector": magnitudes
	}


	return physical_activity_data











