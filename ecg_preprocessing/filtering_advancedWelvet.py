import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pywt
import pandas as pd
import math
import h5py
from tqdm import tqdm

# Parameters
fs = 300
lowcut = 0.5
highcut = 45.0
butter_order = 4
wavelet = 'db2'
decomposition_level = 5

# Paths
input_h5 = "corrected_ecg_signals_before_filtering.h5"
output_file = "allfilteredsignals_ispravni.h5"


def calculate_snr(original, filtered):
    """Calculate SNR between original and filtered signal"""
    noise = original - filtered
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Avoid division by zero
    if noise_power < 1e-10:
        return 50.0  # Cap at reasonable value
    
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return min(snr, 50.0)  # Cap SNR at reasonable maximum


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply Butterworth bandpass filter"""
    try:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Ensure cutoff frequencies are valid
        low = max(low, 0.001)
        high = min(high, 0.999)
        
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y
    except Exception as e:
        print(f"Bandpass filter failed: {e}")
        return data


def safe_wavelet_denoising(data, wavelet, level):
    """Safe wavelet denoising with error handling"""
    try:
        # Ensure signal is long enough for decomposition
        min_length = 2**(level + 1)
        if len(data) < min_length:
            level = max(1, int(np.log2(len(data))) - 2)
        
        coeffs = pywt.wavedec(data, wavelet, level=level, mode='symmetric')
        
        # Zero out first few detail levels (high frequency noise)
        for i in range(1, min(4, len(coeffs))):
            coeffs[i] = np.zeros_like(coeffs[i])
        
        denoised = pywt.waverec(coeffs, wavelet, mode='symmetric')
        
        # Handle length mismatch
        if len(denoised) > len(data):
            denoised = denoised[:len(data)]
        elif len(denoised) < len(data):
            denoised = np.pad(denoised, (0, len(data) - len(denoised)), 'edge')
            
        return denoised
    except Exception as e:
        print(f"Wavelet denoising failed: {e}")
        return data


def visu_shrink(var, coeffs):
    """VisuShrink threshold"""
    N = len(coeffs)
    if N <= 1:
        return 0.1
    return math.sqrt(var) * math.sqrt(2 * math.log(N))


def sure_shrink(var, coeffs):
    """SureShrink threshold"""
    N = len(coeffs)
    if N <= 1:
        return 0.1
        
    sqr_coeffs = [coeff**2 for coeff in coeffs]
    sqr_coeffs.sort()
    
    r = float('inf')
    pos = 0
    
    for idx, sqr_coeff in enumerate(sqr_coeffs):
        remaining = N - (idx + 1)
        if remaining >= 0:
            new_r = (N - 2 * (idx + 1) + remaining * sqr_coeff + sum(sqr_coeffs[:idx + 1])) / N
            if new_r < r:
                r = new_r
                pos = idx
    
    return math.sqrt(var) * math.sqrt(sqr_coeffs[pos])


def heur_sure(var, coeffs):
    """HeurSure threshold selection"""
    N = len(coeffs)
    if N <= 1:
        return 0.1
        
    s = sum(coeff**2 for coeff in coeffs)
    theta = max(0, (s - N) / N)
    miu = (math.log(N, 2)**1.5) / math.sqrt(N)
    
    visu_thresh = visu_shrink(var, coeffs)
    
    if theta < miu:
        return visu_thresh
    else:
        sure_thresh = sure_shrink(var, coeffs)
        return min(visu_thresh, sure_thresh)


def simple_tsd(data, method='heursure', wavelets_name='db2', level=5):
    """Simple threshold-based wavelet denoising"""
    try:
        # Ensure signal is long enough
        min_length = 2**(level + 1)
        if len(data) < min_length:
            level = max(1, int(np.log2(len(data))) - 2)
        
        # Decompose signal
        coeffs = pywt.wavedec(data, wavelets_name, level=level, mode='symmetric')
        
        # Estimate noise variance from first detail level
        if len(coeffs) > 1:
            cD1 = coeffs[1]
            var = (np.median(np.abs(cD1)) / 0.6745)**2
        else:
            var = np.var(data) * 0.1  # Fallback
        
        # Apply thresholding to detail coefficients
        for i in range(1, len(coeffs)):
            if len(coeffs[i]) > 0:
                if method == 'heursure':
                    thre = heur_sure(var, coeffs[i])
                elif method == 'sure':
                    thre = sure_shrink(var, coeffs[i])
                else:  # visu
                    thre = visu_shrink(var, coeffs[i])
                
                # Apply soft thresholding
                coeffs[i] = pywt.threshold(coeffs[i], thre, mode='soft')
        
        # Reconstruct signal
        reconstructed = pywt.waverec(coeffs, wavelets_name, mode='symmetric')
        
        # Handle length mismatch
        if len(reconstructed) > len(data):
            reconstructed = reconstructed[:len(data)]
        elif len(reconstructed) < len(data):
            reconstructed = np.pad(reconstructed, (0, len(data) - len(reconstructed)), 'edge')
            
        return reconstructed
        
    except Exception as e:
        print(f"TSD failed: {e}")
        return data


def faster_ti(data, step=None, method='heursure', wavelets_name='db2', level=5):
    """Translation-invariant wavelet denoising"""
    try:
        # Adaptive step size based on signal length
        if step is None:
            step = max(50, len(data) // 20)
        
        # Limit number of shifts for computational efficiency
        max_shifts = min(5, max(1, len(data) // step))
        
        final_data = np.zeros_like(data, dtype=np.float64)
        
        for i in range(max_shifts):
            shift = i * step
            # Circular shift
            temp_data = np.roll(data, shift)
            
            # Denoise shifted signal
            temp_data = simple_tsd(temp_data, method=method, wavelets_name=wavelets_name, level=level)
            
            # Shift back
            temp_data = np.roll(temp_data, -shift)
            
            # Accumulate
            final_data += temp_data
        
        # Average all shifted versions
        final_data = final_data / max_shifts
        
        return final_data.astype(np.float32)
        
    except Exception as e:
        print(f"TI denoising failed: {e}")
        return data.astype(np.float32)


def enhanced_ecg_filter(signal, fs=300):
    """Enhanced ECG filtering combining multiple techniques"""
    try:
        # Step 1: Light pre-filtering to remove extreme artifacts
        signal_prefiltered = butter_bandpass_filter(signal, 0.1, 100, fs, order=2)
        
        # Step 2: Translation-invariant wavelet denoising
        signal_denoised = faster_ti(signal_prefiltered, method='heursure', wavelets_name='db2', level=4)
        
        # Step 3: Light post-filtering to ensure clean frequency content
        signal_final = butter_bandpass_filter(signal_denoised, lowcut, highcut, fs, order=2)
        
        # Check if filtering was too aggressive
        orig_std = np.std(signal)
        final_std = np.std(signal_final)
        
        if final_std < 0.05 * orig_std:  # Signal was over-filtered
            print("Warning: Over-filtering detected, using lighter approach")
            signal_final = butter_bandpass_filter(signal, 0.5, 50, fs, order=1)
        
        return signal_final
        
    except Exception as e:
        print(f"Enhanced filtering failed: {e}")
        return signal


def process_and_save_filtered_signals():
    """Process all signals with improved error handling and progress tracking"""
    try:
        with h5py.File(input_h5, 'r') as hf:
            signals = [np.array(sig) for sig in hf['signals']]
            labels = [label.decode('ascii') for label in hf['labels']]
            filenames = [name.decode('ascii') for name in hf['filenames']]
            total_records = len(signals)
            
            print(f"\nFound {total_records} signals to process")

        with h5py.File(output_file, 'w') as hf_out:
            dt = h5py.vlen_dtype(np.float32)
            hf_out.create_dataset("filtered_signals", (total_records,), dtype=dt)
            hf_out.create_dataset("original_signals", (total_records,), dtype=dt)
            hf_out.create_dataset("labels", (total_records,), dtype='S1')
            hf_out.create_dataset("filenames", (total_records,), dtype='S100')
            hf_out.create_dataset("snr_before", (total_records,), dtype='f')
            hf_out.create_dataset("snr_after", (total_records,), dtype='f')
            hf_out.attrs['sampling_rate'] = fs

            snr_improvements = []
            successful_processes = 0

            for i in tqdm(range(total_records), desc="Processing signals"):
                try:
                    signal = signals[i].astype(np.float64)
                    
                    # Skip very short signals
                    if len(signal) < 100:
                        print(f"Skipping signal {i}: too short ({len(signal)} samples)")
                        continue
                    
                    # Apply enhanced filtering
                    filtered_signal = enhanced_ecg_filter(signal, fs)
                    
                    # Calculate SNR (approximate)
                    snr_before = 20 * np.log10(np.std(signal) / (np.std(signal - np.mean(signal)) + 1e-10))
                    snr_after = calculate_snr(signal, filtered_signal)
                    
                    # Store results
                    hf_out["filtered_signals"][i] = filtered_signal.astype('float32')
                    hf_out["original_signals"][i] = signal.astype('float32')
                    hf_out["labels"][i] = labels[i].encode('ascii')
                    hf_out["filenames"][i] = filenames[i].encode('ascii')
                    hf_out["snr_before"][i] = snr_before
                    hf_out["snr_after"][i] = snr_after
                    
                    snr_improvements.append(snr_after - snr_before)
                    successful_processes += 1

                except Exception as e:
                    print(f"Error processing signal {i} ({filenames[i] if i < len(filenames) else 'unknown'}): {e}")
                    # Store original signal as fallback
                    if i < len(signals):
                        hf_out["filtered_signals"][i] = signals[i].astype('float32')
                        hf_out["original_signals"][i] = signals[i].astype('float32')
                        if i < len(labels):
                            hf_out["labels"][i] = labels[i].encode('ascii')
                        if i < len(filenames):
                            hf_out["filenames"][i] = filenames[i].encode('ascii')
                        hf_out["snr_before"][i] = 0.0
                        hf_out["snr_after"][i] = 0.0
                    continue

            # Print statistics
            if snr_improvements:
                avg_improvement = np.mean(snr_improvements)
                print(f"\nProcessing completed!")
                print(f"Successfully processed: {successful_processes}/{total_records} signals")
                print(f"Average SNR improvement: {avg_improvement:.2f} dB")
                print(f"SNR improvement range: {np.min(snr_improvements):.2f} to {np.max(snr_improvements):.2f} dB")
            
            print(f"File saved: {output_file}")
            print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"Error in main processing: {e}")
        raise


def plot_sample_comparisons(num_samples=4):
    """Plot sample comparisons with better visualization"""
    try:
        with h5py.File(output_file, 'r') as hf:
            labels = [label.decode('ascii') for label in hf["labels"][:]]
            
            # Find one example of each class
            class_indices = {}
            for i, label in enumerate(labels):
                if label not in class_indices:
                    class_indices[label] = i
                if len(class_indices) >= num_samples:
                    break

            plt.figure(figsize=(16, 12))
            
            for i, (label, idx) in enumerate(class_indices.items()):
                class_name = {
                    'N': 'Normal Sinus Rhythm', 
                    'A': 'Atrial Fibrillation', 
                    'O': 'Other Rhythm', 
                    '~': 'Noisy Signal'
                }.get(label, f'Class {label}')
                
                original = np.array(hf["original_signals"][idx])
                filtered = np.array(hf["filtered_signals"][idx])
                
                # Time axis
                t = np.arange(len(original)) / fs
                
                # Full signal comparison
                ax1 = plt.subplot(len(class_indices), 2, 2*i + 1)
                ax1.plot(t, original, 'b-', label='Original', alpha=0.7, linewidth=0.8)
                ax1.plot(t, filtered, 'r-', label='Filtered', alpha=0.9, linewidth=1)
                ax1.set_title(f'{class_name} - Full Signal', fontsize=11)
                ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Amplitude')
                
                # Zoomed view (first 3 seconds or available length)
                ax2 = plt.subplot(len(class_indices), 2, 2*i + 2)
                zoom_samples = min(int(3 * fs), len(original))
                t_zoom = t[:zoom_samples]
                
                ax2.plot(t_zoom, original[:zoom_samples], 'b-', label='Original', alpha=0.7, linewidth=1)
                ax2.plot(t_zoom, filtered[:zoom_samples], 'r-', label='Filtered', alpha=0.9, linewidth=1.2)
                ax2.set_title(f'{class_name} - Zoomed (3s)', fontsize=11)
                ax2.legend(fontsize=9)
                ax2.grid(True, alpha=0.3)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Amplitude')
                
                # Add SNR info if available
                try:
                    snr_before = hf["snr_before"][idx]
                    snr_after = hf["snr_after"][idx]
                    ax2.text(0.02, 0.98, f'SNR: {snr_before:.1f}→{snr_after:.1f} dB', 
                            transform=ax2.transAxes, va='top', ha='left',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except:
                    pass

            plt.tight_layout()
            plt.suptitle('Enhanced ECG Filtering Results', y=0.98, fontsize=14)
            plt.show()
            
    except Exception as e:
        print(f"Error in plotting: {e}")


def analyze_filtering_performance():
    """Analyze overall filtering performance"""
    try:
        with h5py.File(output_file, 'r') as hf:
            snr_before = np.array(hf["snr_before"][:])
            snr_after = np.array(hf["snr_after"][:])
            labels = [label.decode('ascii') for label in hf["labels"][:]]
            
            # Remove invalid SNR values
            valid_mask = (snr_before > -50) & (snr_after > -50) & (snr_before < 50) & (snr_after < 50)
            snr_before = snr_before[valid_mask]
            snr_after = snr_after[valid_mask]
            labels = np.array(labels)[valid_mask]
            
            if len(snr_before) == 0:
                print("No valid SNR measurements found")
                return
            
            print("\n" + "="*50)
            print("FILTERING PERFORMANCE ANALYSIS")
            print("="*50)
            
            improvement = snr_after - snr_before
            print(f"Overall SNR improvement: {np.mean(improvement):.2f} ± {np.std(improvement):.2f} dB")
            print(f"Median SNR improvement: {np.median(improvement):.2f} dB")
            print(f"Success rate: {np.sum(improvement > 0) / len(improvement) * 100:.1f}%")
            
            # Per-class analysis
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                if np.sum(mask) > 0:
                    class_improvement = improvement[mask]
                    class_name = {
                        'N': 'Normal', 'A': 'AF', 'O': 'Other', '~': 'Noise'
                    }.get(label, label)
                    print(f"{class_name}: {np.mean(class_improvement):.2f} ± {np.std(class_improvement):.2f} dB ({np.sum(mask)} signals)")
            
    except Exception as e:
        print(f"Error in performance analysis: {e}")


if __name__ == "__main__":
    try:
        print("Starting enhanced ECG filtering pipeline...")
        process_and_save_filtered_signals()
        plot_sample_comparisons()
        analyze_filtering_performance()
        print("\nProcessing completed successfully!")
        
    except KeyboardInterrupt:
        print("\nProcessing stopped by user")
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()