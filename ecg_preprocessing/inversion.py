import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import h5py
from matplotlib.patches import Rectangle

# Paths
data_dir = '/Users/adnaherak/Downloads/af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0/training2017'
reference_csv = os.path.join(data_dir, "REFERENCE.csv")
output_h5 = "corrected_ecg_signals_before_filtering.h5"

def plot_ecg_with_window_analysis(signal, title, window_start=1000, window_size=600):
    """Plot ECG with proper midpoint/mean relationship"""
    plt.figure(figsize=(10, 4))
    
    window = signal[window_start:window_start+window_size]
    window_min = np.min(window)
    window_max = np.max(window)
    midpoint = (window_min + window_max)/2
    global_mean = np.mean(signal)
    
    # Plot signal and features
    plt.plot(signal, 'b-', alpha=0.7)
    plt.axvspan(window_start, window_start+window_size, color='yellow', alpha=0.3)
    
    # Critical fix: Show proper midpoint/mean relationship
    plt.hlines(midpoint, window_start, window_start+window_size, 
               colors='red', linestyles='dashed', linewidth=2, label=f'Midpoint ({midpoint:.2f})')
    plt.hlines(global_mean, 0, len(signal), colors='green', linestyles=':', 
               label=f'Global Mean ({global_mean:.2f})')
    
    # Add diagnostic text
    relationship = "ABOVE" if midpoint > global_mean else "BELOW"
    plt.text(window_start+50, midpoint+0.2, 
             f"Midpoint is {relationship} global mean",
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(window_start-200, window_start+window_size+200)
    plt.show()
def find_demonstration_examples():
    """Find examples that strictly follow the paper's rules"""
    df = pd.read_csv(reference_csv, header=None, names=['filename', 'label'])
    normal_example = None
    inverted_example = None
    
    for filename in tqdm(df['filename'], desc="Finding strict examples"):
        signal = wfdb.rdrecord(os.path.join(data_dir, filename)).p_signal[:, 0]
        global_mean = np.mean(signal)
        
        # Analyze multiple windows for robust detection
        window_size = 600
        n_windows = min(5, len(signal)//window_size)  # Check first 5 windows
        midpoints = []
        
        for i in range(n_windows):
            window = signal[i*window_size:(i+1)*window_size]
            midpoints.append((np.min(window)+np.max(window))/2)
        
        avg_midpoint = np.mean(midpoints)
        
        # Strict selection criteria
        if avg_midpoint > global_mean + 0.5 and normal_example is None:
            normal_example = (filename, signal)
        elif avg_midpoint < global_mean - 0.5 and inverted_example is None:
            inverted_example = (filename, signal)
        
        if normal_example and inverted_example:
            break
    
    return normal_example, inverted_example

def plot_inversion_detection_examples():
    """Create figure showing window analysis for normal/inverted ECGs"""
    normal_example, inverted_example = find_demonstration_examples()
    
    if normal_example:
        filename, signal = normal_example
        plot_ecg_with_window_analysis(signal, f"(a) Non-Inverted ECG Record\n{filename}")
    
    if inverted_example:
        filename, signal = inverted_example
        plot_ecg_with_window_analysis(signal, f"(b) Inverted ECG Record\n{filename}")
    
    if not normal_example or not inverted_example:
        print("Warning: Could not find both normal and inverted examples")

# --------------------------------------------------
# Main Processing (unchanged)
# --------------------------------------------------
def load_raw_signals():
    """Load all WFDB signals and labels from the dataset."""
    df = pd.read_csv(reference_csv, header=None, names=['filename', 'label'])
    signals = []
    labels = []
    
    for filename in tqdm(df['filename'], desc="Loading signals"):
        record = wfdb.rdrecord(os.path.join(data_dir, filename))
        signals.append(record.p_signal[:, 0])  # First channel only
        labels.append(df[df['filename'] == filename]['label'].values[0])
    
    return signals, labels, df['filename'].values

def detect_inverted_signal(signal, window_size=600, threshold=0.6):
    """Detect if ECG is inverted using sliding-window midpoints."""
    signal_mean = np.mean(signal)
    n_windows = len(signal) // window_size
    below_mean = 0
    
    for i in range(n_windows):
        window = signal[i*window_size : (i+1)*window_size]
        midpoint = (np.min(window) + np.max(window)) / 2
        if midpoint < signal_mean:
            below_mean += 1
    
    return (below_mean / n_windows) > threshold  # True if inverted

def correct_inverted_signal(signal):
    """Flip the signal vertically."""
    return -signal

def process_and_save_corrected():
    signals, labels, filenames = load_raw_signals()
    n_inverted = 0
    
    with h5py.File(output_h5, 'w') as hf:
        # Create datasets for variable-length signals
        dt = h5py.vlen_dtype(np.float32)
        hf.create_dataset("signals", (len(signals),), dtype=dt)
        hf.create_dataset("labels", (len(labels),), dtype='S1')
        hf.create_dataset("filenames", (len(filenames),), dtype='S100')
        
        for i, signal in enumerate(tqdm(signals, desc="Processing")):
            # Detect and correct inversion
            if detect_inverted_signal(signal):
                signal = correct_inverted_signal(signal)
                n_inverted += 1
            
            # Save to HDF5
            hf["signals"][i] = signal.astype('float32')
            hf["labels"][i] = labels[i].encode('ascii')
            hf["filenames"][i] = filenames[i].encode('ascii')
    
    print(f"\nCorrected {n_inverted}/{len(signals)} inverted signals.")
    print(f"Saved to {output_h5}")

if __name__ == "__main__":
    # First plot demonstration examples from raw data
    plot_inversion_detection_examples()
    
    # Then process all signals
    process_and_save_corrected()