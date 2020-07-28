"""

Bone Health Algorithm

"""


def compute_intensity(fft_magnitudes, sampling_frequency, high_cut_off, fftpoints=512):
    OI = 0
    fs = sampling_frequency
    fc = high_cut_off
    kc = int((fftpoints/fs)* fc) + 1

    magnitudes = fft_magnitudes


    f = []
    for i in range(0, int(fftpoints/2)+1):
        f.append((fs*i)/fftpoints)

    for k in range(0, kc):
        OI = OI + (magnitudes[k] * f[k])

    return OI


def compute_impact(intensity):
    if(intensity > 10):
    	return True
    else: 
    	return False