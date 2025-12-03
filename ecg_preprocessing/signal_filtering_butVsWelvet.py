import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt
import pywt
import pandas as pd
import math

# Parameters with adjusted values for better performance
fs = 300
lowcut = 0.5
highcut = 45.0
butter_order = 4
wavelet = 'sym8'
decomposition_level = 5  # Reduced from 9 to 5 for faster processing
max_signals_per_class = 2  # Process fewer signals for testing

# Paths
data_dir = '/Users/adnaherak/Downloads/af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0/training2017'
reference_csv = os.path.join(data_dir, "REFERENCE.csv")

def load_labels():
    df = pd.read_csv(reference_csv, header=None, names=['filename', 'label'])
    return df

def load_ecg_signal(filename):
    try:
        record = wfdb.rdrecord(os.path.join(data_dir, filename))
        signal = record.p_signal[:, 0]
        return signal  # Return the full signal without length check
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def safe_wavelet_denoising(data, wavelet, level):
    """Wrapper with error handling for wavelet denoising"""
    try:
        coeffs = pywt.wavedec(data, wavelet, level=level)
        for i in range(1, min(4, len(coeffs))):  # Fixed parenthesis here
            coeffs[i] = np.zeros_like(coeffs[i])
        denoised = pywt.waverec(coeffs, wavelet)
        return denoised[:len(data)]
    except Exception as e:
        print(f"Wavelet denoising failed: {e}")
        return data  # Return original if denoising fails

def heur_sure(var, coeffs):
    N = len(coeffs)
    s = sum(math.pow(coeff, 2) for coeff in coeffs)
    theta = (s - N) / N
    miu = math.pow(math.log2(N), 3/2) / math.pow(N, 1/2)
    if theta < miu:
        return visu_shrink(var, coeffs)
    else:
        return min(visu_shrink(var, coeffs), sure_shrink(var, coeffs))

def visu_shrink(var, coeffs):
    N = len(coeffs)
    return math.sqrt(var) * math.sqrt(2 * math.log(N))

def sure_shrink(var, coeffs):
    N = len(coeffs)
    sqr_coeffs = [math.pow(coeff, 2) for coeff in coeffs]
    sqr_coeffs.sort()
    r = 0
    pos = 0
    for idx, sqr_coeff in enumerate(sqr_coeffs):
        new_r = (N - 2 * (idx + 1) + (N - (idx + 1)) * sqr_coeff + sum(sqr_coeffs[:idx+1])) / N
        if r == 0 or r > new_r:
            r = new_r
            pos = idx
    return math.sqrt(var) * math.sqrt(sqr_coeffs[pos])

def faster_ti(data, step=200, method='heursure', wavelets_name='sym5', level=5):
    """Optimized translation-invariant denoising with fewer shifts"""
    try:
        num = min(5, math.ceil(len(data)/step))  # Limit to 5 shifts
        final_data = np.zeros_like(data)
        
        for i in range(num):
            temp_data = np.roll(data, i*step)
            temp_data = simple_tsd(temp_data, method=method, wavelets_name=wavelets_name, level=level)
            temp_data = np.roll(temp_data, -i*step)
            final_data += temp_data[:len(data)]
        
        return final_data / num
    except Exception as e:
        print(f"TI denoising failed: {e}")
        return data

def simple_tsd(data, method='heursure', wavelets_name='sym5', level=5):
    """Simplified threshold shrinkage denoising"""
    try:
        coeffs = pywt.wavedec(data, wavelets_name, level=level)
        cA, cD = pywt.dwt(data, wavelets_name)
        var = np.median(np.abs(cD)) / 0.6745
        
        for i in range(1, len(coeffs)):
            if method == 'heursure':
                thre = heur_sure(var, coeffs[i])
            coeffs[i] = pywt.threshold(coeffs[i], thre, mode='soft')
        
        return pywt.waverec(coeffs, wavelets_name)[:len(data)]
    except Exception as e:
        print(f"TSD failed: {e}")
        return data

def process_and_plot_by_class():
    labels_df = load_labels()
    class_signals = {'N': [], 'A': [], 'O': [], '~': []}
    
    for _, row in labels_df.iterrows():
        if all(len(v) >= max_signals_per_class for v in class_signals.values()):
            break  # Limit number of signals per class
            
        filename = row['filename']
        label = row['label']
        signal = load_ecg_signal(filename)
        
        if signal is not None:
            try:
                butter_filtered = butter_bandpass_filter(signal, lowcut, highcut, fs, butter_order)
                wavelet_simple = safe_wavelet_denoising(signal, wavelet, decomposition_level)
                wavelet_advanced = faster_ti(signal)
                
                class_signals[label].append({
                    'original': signal,
                    'butterworth': butter_filtered,
                    'wavelet_simple': wavelet_simple,
                    'wavelet_advanced': wavelet_advanced
                })
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    plot_results(class_signals)

def plot_results(class_signals):
    for label, signals in class_signals.items():
        if signals:
            fig, axes = plt.subplots(4, 1, figsize=(15, 10))
            titles = ['Original', 'Butterworth', 'Simple Wavelet', 'Advanced Wavelet']
            
            for ax, title, key in zip(axes, titles, ['original', 'butterworth', 'wavelet_simple', 'wavelet_advanced']):
                ax.plot(signals[0][key])
                ax.set_title(f'{title} ({label})')
                ax.grid()
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    try:
        process_and_plot_by_class()
    except KeyboardInterrupt:
        print("\nProcessing stopped by user")
    except Exception as e:
        print(f"Error in main execution: {e}")