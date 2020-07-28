"""

Accelerometer simulation

"""
import numpy as np

def create3AxisAccelData(
	sampling_rate,
	duration,
	add_noise=False
):

	# duration is in milliseconds?

	time = np.arange(0, duration*sampling_rate+1) * 10  # times in milliseconds
	np.random.seed(123)
	# generate some noisy walking data
	x = 2*np.sin(time/30) + np.random.normal(0.5, 0.4, len(time))  # ML medio-lateral -> transverse rotation around the vertical axis
	y = 4*np.sin(time/80) + np.random.normal(1.0, 0.5, len(time))  # VT vertical -> hip hiking around the sagitall axis
	z = 3*np.sin(time/90) + np.random.normal(0.0, 0.4, len(time))  # AP antero-posterior -> tilt around the frontal axis

	if(add_noise):
		# Adjust amount of movement at various timepoints
		signals = [x, y, z]

		for s in signals:
		    s[20000:40000] = s[20000:40000] * 10
		    s[40000:] = s[40000:] * 50

	return time,x,y,z


