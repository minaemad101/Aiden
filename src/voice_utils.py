import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES//HOP_LENGTH  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = SAMPLE_RATE// HOP_LENGTH  # 10ms per audio frame
TOKENS_PER_SECOND = SAMPLE_RATE// N_SAMPLES_PER_TOKEN  # 20ms per audio token

def normalize(audio_data):
  max_amplitude = np.max(np.abs(audio_data))
  normalized_data = audio_data / max_amplitude
  return normalized_data

def filtering(audio_data, sampling_rate):
    """
    Filters an audio signal with a high-pass Butterworth filter.

    Parameters
    ----------
    audio_data : ndarray
        The input audio signal to be filtered.
    sampling_rate : int
        The sampling rate of the audio signal, in Hz.

    Returns
    -------
    filtered_data : ndarray
        The filtered audio signal.
    """
    # Define the high-pass filter parameters
    cutoff_freq = 500  # in Hz
    nyquist_freq = sampling_rate / 2
    filter_order = 4

    # Calculate the filter coefficients
    wc = 2 * np.pi * cutoff_freq / sampling_rate
    b, a = butterworth_highpass(wc, filter_order)
    #print(b,a)

    # Apply the high-pass filter to the audio data
    filtered_data = apply_filter(b, a, audio_data)

    return filtered_data


def butterworth_highpass(wc, order):
    # Calculate the poles of the filter
    poles = wc * np.exp(1j * (2 * np.arange(1, order+1) + order - 1) * np.pi / (2 * order))

    # Calculate the numerator coefficients of the transfer function
    b = np.zeros(order+1)
    b[order] = 1

    # Calculate the denominator coefficients of the transfer function
    a = np.concatenate(([1], -np.real(poles)))

    # Normalize the coefficients so that a[0] = 1
    b = b / a[0]
    a = a / a[0]

    return b, a


def apply_filter(b, a, x):
    # Apply the filter using difference equations
    y = np.zeros_like(x)
    for n in range(len(x)):
        y[n] = b[0] * x[n]
        for k in range(1, len(b)):
            if n - k >= 0:
                y[n] += b[k] * x[n-k] - a[k] * y[n-k]

    return y

def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    
    Parameters
    ----------
    array: ndarray
      The audio array to pad or trim.
    length: int
      The desired length of the audio array. Defaults to N_SAMPLES.
    axis: int 
      The axis along which to pad or trim the audio array. Defaults to -1.
    
    Returns
    -------
    array: ndarray
      padded or trimmed audio array with the specified length
    """
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array

def window_fn(n):
    """Compute the Hanning window function of length n."""
    return 0.5 - 0.5 * np.cos(2 * np.pi / n * (np.arange(n) + 0.5))

def rfft(x, n=None, axis=-1):
    """
    Compute the real-valued FFT of a 1D or 2D array using numpy's rfft function.

    Parameters
    ----------
    x : array_like
        Input array. If x is real, it should be a 1D or 2D array of shape (..., n) or (..., n_samples, n_channels).
        If x is complex, it should be a 2D array of shape (..., n, 2).
    n : int, optional
        FFT size. If not specified, the FFT size is set to the length of the last dimension of x.
    axis : int, optional
        Axis along which to compute the FFT.

    Returns
    -------
    y : ndarray
        The real-valued FFT of x, with dimensions (..., n // 2 + 1) or (..., n_samples, n_channels // 2 + 1).
    """
    # Check if x is real or complex
    is_complex = np.iscomplexobj(x)

    # Get the length of the FFT
    if n is None:
        n = x.shape[axis]
    elif n < x.shape[axis]:
        raise ValueError("FFT size must be greater than or equal to input size along the given axis.")

    # Compute the FFT
    y = np.fft.rfft(x, n=n, axis=axis)

    # If x is complex, convert y to a complex array
    if is_complex:
        y = y[..., 0] + 1j * y[..., 1]

    return y

def stft(x, nperseg=400, noverlap=200, nfft=400):
    """
    Compute the Short-Time Fourier Transform (STFT) of a signal.

    Parameters
    ----------
    x : array_like
        Input signal.
    nperseg : int, optional
        Length of each segment. Defaults to 256.
    noverlap : int, optional
        Number of points to overlap between segments. Defaults to nperseg // 2.
    nfft : int, optional
        Length of the FFT. If not specified, nfft = nperseg.

    Returns
    -------
    Zxx : ndarray
        STFT of x, with dimensions (frequencies, time).
    """
    # Check inputs
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input signal must be 1-dimensional.")


    # Compute hanning window
    window = window_fn(nperseg)

    # Compute the STFT
    hopsize = nperseg - noverlap
    nframes = 1 + (len(x) - nperseg) // hopsize
    padded_len = nperseg + (nframes - 1) * hopsize
    x_pad = np.zeros(padded_len)
    x_pad[:len(x)] = x
    Xs = np.tile(np.arange(nperseg), (nframes, 1)) + np.tile(np.arange(0, nframes * hopsize, hopsize), (nperseg, 1)).T
    X = x_pad[Xs]
    X = X * window
    Zxx = rfft(X, n=nfft, axis=1)

    return Zxx

def mel_filterbank(sr, n_fft, n_mels, fmin=0, fmax=8000):
    """
    Compute a Mel filterbank.

    Parameters
    ----------
    sr : int
        Sampling rate of the audio signal.
    n_fft : int
        Number of FFT bins.
    n_mels : int
        Number of Mel filterbanks.
    fmin : float
        Minimum frequency for the Mel filterbank.
    fmax : float
        Maximum frequency for the Mel filterbank.

    Returns
    -------
    filterbank : ndarray, shape (n_mels, n_fft//2 + 1)
        Mel filterbank matrix.
    """
    # Convert minimum and maximum frequencies to Mel scale
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)

    # Generate Mel scale vector
    mel_scale = np.linspace(mel_min, mel_max, num=n_mels+2)

    # Convert Mel scale back to linear frequency scale
    hz_scale = 700 * (10**(mel_scale / 2595) - 1)

    # Convert frequency to FFT bin index
    bin_scale = np.floor((n_fft + 1) * hz_scale / sr)

    # Define the filterbank matrix
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))

    # Compute the filterbank matrix
    for m in range(1, n_mels + 1):
        f_m_minus = int(bin_scale[m - 1])
        f_m = int(bin_scale[m])
        f_m_plus = int(bin_scale[m + 1])

        for k in range(f_m_minus, f_m):
            filterbank[m - 1, k] = (k - bin_scale[m - 1]) / (bin_scale[m] - bin_scale[m - 1])
        for k in range(f_m, f_m_plus):
            filterbank[m - 1, k] = (bin_scale[m + 1] - k) / (bin_scale[m + 1] - bin_scale[m])

    return filterbank

def log_mel_feature(audio,sr):
    n_aud = normalize(audio)
    n_aud = filtering(n_aud,sr)
    audio = pad_or_trim(audio)

    D = np.zeros((N_FFT // 2 + 1,N_FRAMES), dtype=np.complex64)

    for i, start in enumerate(range(0, audio.shape[0] - N_FFT, HOP_LENGTH)):
        window = window_fn(N_FFT)
        frame = audio[start:start + N_FFT] * window
        D[:, i] = stft(frame)

    magnitude = np.abs(D)
    #log_amplitude = 20 * np.log10(magnitude)
    power = magnitude**2
    mel_basis = mel_filterbank(sr=sr, n_fft = N_FFT, n_mels=N_MELS)
    mel_spec = np.dot(mel_basis, power)+ 1e-6
    log_mel_spec = 20 * np.log10(mel_spec)
    x_min = np.min(log_mel_spec)
    x_max = np.max(log_mel_spec)
    x_norm = 2 * (log_mel_spec - x_min) / (x_max - x_min) - 1
    return x_norm