import h5py
import numpy as np
from scipy.signal import resample
import os
import matplotlib.pyplot as plt  # <-- Missing import added
# Configuration
INPUT_H5_FILE = "wavelet_filtered_ecg_signals.h5"
OUTPUT_H5_FILE = "resampled_ecg_signals.h5"
ORIGINAL_SAMPLE_RATE = 300  # Hz
TARGET_SAMPLE_RATE = 150    # Hz
RESAMPLE_RATIO = TARGET_SAMPLE_RATE / ORIGINAL_SAMPLE_RATE

def resample_signal(signal, original_rate, target_rate):
    """Resample signal using Fourier method"""
    num_samples = int(len(signal) * target_rate / original_rate)
    return resample(signal, num_samples)

def process_resampling():
    # Load the original data
    with h5py.File(INPUT_H5_FILE, 'r') as hf:
        filtered_signals = hf['filtered_signals'][:]
        labels = hf['labels'][:]
        filenames = hf['filenames'][:]
        
        original_length = filtered_signals.shape[1]
        new_length = int(original_length * RESAMPLE_RATIO)
        
        print(f"Original sampling rate: {ORIGINAL_SAMPLE_RATE} Hz")
        print(f"Target sampling rate: {TARGET_SAMPLE_RATE} Hz")
        print(f"Resampling from {original_length} to {new_length} samples per signal")
        
        # Create new HDF5 file for resampled data
        with h5py.File(OUTPUT_H5_FILE, 'w') as new_hf:
            # Create datasets with same structure but shorter signals
            new_hf.create_dataset("filtered_signals", 
                                (len(filtered_signals), new_length), 
                                dtype='float32')
            new_hf.create_dataset("labels", data=labels)
            new_hf.create_dataset("filenames", data=filenames)
            
            # Process each signal
            for i in range(len(filtered_signals)):
                resampled_signal = resample_signal(
                    filtered_signals[i], 
                    ORIGINAL_SAMPLE_RATE, 
                    TARGET_SAMPLE_RATE
                )
                new_hf["filtered_signals"][i] = resampled_signal.astype('float32')
                
                if (i+1) % 500 == 0:
                    print(f"Processed {i+1}/{len(filtered_signals)} signals...")
    
    print(f"\nResampling complete! Results saved to {OUTPUT_H5_FILE}")
    print(f"New file size: {os.path.getsize(OUTPUT_H5_FILE)/1024/1024:.2f} MB")

def verify_resampling():
    """Verify the resampling by plotting a sample"""
    with h5py.File(OUTPUT_H5_FILE, 'r') as hf:
        resampled_signal = hf['filtered_signals'][0]
        label = hf['labels'][0].decode('ascii')
    
    with h5py.File(INPUT_H5_FILE, 'r') as hf:
        original_signal = hf['filtered_signals'][0]
    
    # Create time axes
    orig_time = np.arange(len(original_signal)) / ORIGINAL_SAMPLE_RATE
    resamp_time = np.arange(len(resampled_signal)) / TARGET_SAMPLE_RATE
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(orig_time, original_signal, 'b-', alpha=0.7, label='Original (300 Hz)')
    plt.plot(resamp_time, resampled_signal, 'r-', alpha=0.7, label=f'Resampled ({TARGET_SAMPLE_RATE} Hz)')
    plt.title(f"Resampling Verification ({label} Rhythm)", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_resampling()
    verify_resampling()