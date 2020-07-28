import lib.utils as ut
from scipy.signal import butter, filtfilt, lfilter
import numpy as np
import lib.plot as p
import matplotlib.pyplot as plt

def compute_fft_mag(data, points=512):
    fft = np.fft.fft(data, n=points)
    mag = np.abs(fft)/(points/2)
    return mag


def build_filter(frequency, sample_rate, filter_type, filter_order):
    """
    Build a butterworth filter with specified parameters.
    Calculates the Nyquist frequency and associated frequency cutoff, and
    builds a Butterworth filter from the parameters.
    Parameters
    ----------
    frequency : int or tuple of ints
        The cutoff frequency for the filter. If `filter_type` is set as
        'bandpass' then this needs to be a tuple of integers representing
        the lower and upper bound frequencies. For example, for a bandpass
        filter with range of 2Hz and 10Hz, you would pass in the tuple (2, 10).
        For filter types with a single cutoff frequency then a single integer
        should be used.
    sample_rate : float
        Sampling rate of the signal.
    filter_type : {'lowpass', 'highpass', 'bandpass', 'bandstop'}
        Type of filter to build.
    filter_order: int, optional
        Order of the filter.
    Returns
    -------
    b : ndarray
        Numerator polynomials of the IIR filter.
    a : ndarray
        Denominator polynomials of the IIR filter.
    """

    nyq = 0.5 * sample_rate

    if filter_type == "bandpass":
        nyq_cutoff = (frequency[0] / nyq, frequency[1] / nyq)
    else:
        nyq_cutoff = frequency / nyq

    b, a = butter(filter_order, nyq_cutoff, btype=filter_type, analog=False)

    return b, a


def filter_signal(b, a, signal, filter):
    """
    Filter a signal.
    Simple wrapper around :func:`scipy.signal.filtfilt` to apply a
    foward-backward filter to preserve phase of the input. Requires the
    numerator and denominator polynomials from
    :func:`sensormotion.signal.build_filter`.
    Parameters
    ----------
    b : ndarray
        Numerator polynomial coefficients of the filter.
    a : ndarray
        Denominator polynomial coefficients of the filter.
    signal : ndarray
        Input array to be filtered.
    Returns
    -------
    signal_filtered : ndarray
        Filtered output of the original input signal.
    """
    if(filter=="lfilter"):
        return lfilter(b, a, signal)
    elif(filter=="filtfilt"):
        return filtfilt(b, a, signal)


def fft(signal, sampling_rate, plot=False, show_grid=True, fig_size=(10, 5)):
    """
    Perform FFT on signal.
    Compute 1D Discrete Fourier Transform using Fast Fourier Transform.
    Optionally, plot the power spectrum of the frequency domain.
    Parameters
    ----------
    signal : ndarray
        Input array to be transformed.
    sampling_rate : float
        Sampling rate of the input signal.
    plot : bool, optional
        Toggle to display a plot of the power spectrum.
    show_grid : bool, optional
        If creating a plot, toggle to show grid lines on the figure.
    fig_size : tuple, optional
        If plotting, set the width and height of the resulting figure.
    Returns
    -------
    signal_fft : ndarray
        Transformation of the original input signal.
    """

    n = len(signal)
    t = 1.0 / sampling_rate
    time = range(n)  # Time vector


    # for plotting
    xf = np.linspace(0.0, 1.0 / (2.0 * t), n // 2)

    yf = np.fft.fft(signal) / n  # FFT and normalize

    mag = np.abs(yf)

    if plot:
        f, axarr = plt.subplots(2, 1, figsize=fig_size)

        axarr[0].plot(time, signal)
        axarr[0].set_xlim(min(time), max(time))
        axarr[0].set_xlabel("Time Steps")
        axarr[0].set_ylabel("Amplitude")
        axarr[0].grid(show_grid)

        axarr[1].plot(xf, abs(yf[0 : n // 2]), "r")  # Plot the spectrum
        axarr[1].set_xlabel("Freq (Hz)")
        axarr[1].set_ylabel("|Y(freq)|")
        axarr[1].grid(show_grid)

        f.subplots_adjust(hspace=0.5)
        plt.suptitle("Power Spectrum", size=16)
        plt.show()

    return yf

def vector_magnitude(*args):
    """
    Calculate the vector magnitude/euclidean norm of multiple vectors.
    Given an arbitrary number of input vectors, calculate the vector
    magnitude/euclidean norm using the Pythagorean theorem.
    Parameters
    ----------
    *args : ndarray
        Each parameter is a numpy array representing a single vector. Multiple
        vectors can be passed in, for example, `vector_magnitude(x, y, z)`
    Returns
    -------
    vm : ndarray
        Vector magnitude across all input vectors.
    """
    #print(args)

    n = len(args[0])
    assert all(len(x) == n for x in args), "Vectors have different lengths"

    vm = np.sqrt(sum(x ** 2 for x in args))

    return vm


def rectify_signal(
    signal, rectifier_type="full", plot=False, show_grid=True, fig_size=(10, 5)
):
    """
    Rectify a signal.
    Run a signal through a full or half-wave rectifier. Optionally plot the
    resulting signal.
    Parameters
    ----------
    signal : ndarray
        Input signal to be rectified.
    rectifier_type : {'full', 'half'}, optional
        Type of rectifier to use. Full-wave rectification turns all negative
        values into positive ones. Half-wave rectification sets all negative
        values to zero.
    plot : bool, optional
        Toggle to display a plot of the rectified signal.
    show_grid : bool, optional
        If creating a plot, toggle to show grid lines on the figure.
    fig_size : tuple, optional
        If plotting, set the width and height of the resulting figure.
    Returns
    -------
    output : ndarray
        Rectified signal.
    """

    if rectifier_type == "half":
        output = signal * (signal > 0)
    elif rectifier_type == "full":
        output = np.abs(signal)

    if plot:
        f, ax = plt.subplots(1, 1, figsize=fig_size)

        time = np.arange(len(signal))

        ax.plot(time, signal, color="k", linewidth=1, alpha=0.5, label="Original")
        ax.plot(time, output, color="r", linewidth=0.9, label="Rectified")
        ax.set_xlim(min(time), max(time))
        ax.grid(show_grid)
        ax.legend()

        plt.suptitle("Rectified Signal ({})".format(rectifier_type), size=16)
        plt.show()

    return output

def xcorr(x, y, scale="none", plot=False, show_grid=True, fig_size=(10, 5)):
    """
    Cross-correlation between two 1D signals.
    Calculate the cross-correlation between two signals for all time lags
    (forwards and backwards). If the inputs are different lengths, zeros will
    be appended to the shorter input.
    All 4 scaling options (`none`, `biased`, `unbiased`, and `coeff`)
    reproduce the output from MATLAB's `xcorr()` function.
    Optionally, plots can be created to visualize the cross-correlation values
    at each lag.
    Parameters
    ----------
    x : ndarray
        First input signal.
    y : ndarray
        Second input signal. Pass in `x` again for autocorrelation.
    scale : {'none', 'biased', 'unbiased', 'coeff'}, optional
        Scaling options for the cross-correlation values. Replicates MATLAB's
        options for scaling.
    plot : bool, optional
        Toggle to display a plot of the cross-correlations.
    show_grid : bool, optional
        If creating a plot, toggle to show grid lines on the figure.
    fig_size : tuple, optional
        If plotting, set the width and height of the resulting figure.
    Returns
    -------
    corr : ndarray
        Cross-correlation values.
    lags : ndarray
        Lags for the cross-correlations.
    """

    x = np.array(x)
    y = np.array(y)

    # Pad shorter array if signals are different lengths
    if x.size > y.size:
        pad_amount = x.size - y.size
        y = np.append(y, np.repeat(0, pad_amount))
    elif y.size > x.size:
        pad_amount = y.size - x.size
        x = np.append(x, np.repeat(0, pad_amount))

    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-(x.size - 1), x.size)

    # Scale the correlation values
    # Equivalent to xcorr scaling options in MATLAB
    if scale == "biased":
        corr = corr / x.size
    elif scale == "unbiased":
        corr /= x.size - abs(lags)
    elif scale == "coeff":
        corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    if plot:
        p.plot_signal(
            lags,
            corr,
            title="Cross-correlation (scale: {})".format(scale),
            xlab="Lag",
            ylab="Correlation",
            show_grid=show_grid,
            fig_size=fig_size,
        )

    return corr, lags