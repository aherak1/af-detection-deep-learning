import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Konfiguracija
INPUT_H5_FILE = "resampled_ecg_signals.h5"
OUTPUT_H5_FILE = "normalized_ecg_signals.h5"

def min_max_normalize(signal):
    """Normalizacija signala u opseg [0, 1] koristeći Min-Max skaliranje"""
    min_val = np.min(signal)
    max_val = np.max(signal)
    return (signal - min_val) / (max_val - min_val + 1e-8)  # Mali epsilon za izbjegavanje dijeljenja s nulom

def process_normalization():
    # Učitavanje podataka
    with h5py.File(INPUT_H5_FILE, 'r') as hf:
        resampled_signals = hf['filtered_signals'][:]
        labels = hf['labels'][:]
        filenames = hf['filenames'][:]
        
        print(f"Učitano {len(resampled_signals)} signala za normalizaciju")
        
        # Kreiranje nove HDF5 datoteke
        with h5py.File(OUTPUT_H5_FILE, 'w') as new_hf:
            # Kreiranje dataset-a
            new_hf.create_dataset("normalized_signals", 
                                shape=resampled_signals.shape, 
                                dtype='float32')
            new_hf.create_dataset("labels", data=labels)
            new_hf.create_dataset("filenames", data=filenames)
            
            # Čuvanje parametara normalizacije
            new_hf.create_dataset("original_mins", (len(resampled_signals),), dtype='float32')
            new_hf.create_dataset("original_maxs", (len(resampled_signals),), dtype='float32')
            
            # Procesiranje svakog signala
            for i in range(len(resampled_signals)):
                signal = resampled_signals[i]
                min_val = np.min(signal)
                max_val = np.max(signal)
                
                normalized_signal = min_max_normalize(signal)
                
                new_hf["normalized_signals"][i] = normalized_signal.astype('float32')
                new_hf["original_mins"][i] = min_val
                new_hf["original_maxs"][i] = max_val
                
                if (i+1) % 500 == 0:
                    print(f"Procesirano {i+1}/{len(resampled_signals)} signala...")
    
    print(f"\nNormalizacija završena! Rezultati sačuvani u {OUTPUT_H5_FILE}")
    print(f"Veličina nove datoteke: {os.path.getsize(OUTPUT_H5_FILE)/1024/1024:.2f} MB")

def verify_normalization():
    """Provjera normalizacije kroz grafički prikaz"""
    with h5py.File(OUTPUT_H5_FILE, 'r') as hf:
        normalized_signal = hf['normalized_signals'][0]
        label = hf['labels'][0].decode('ascii')
    
    with h5py.File(INPUT_H5_FILE, 'r') as hf:
        original_signal = hf['filtered_signals'][0]
    
    # Vremenska osa (150 Hz)
    time = np.arange(len(normalized_signal)) / 150
    
    # Mapiranje oznaka ritma na bosanski
    rhythm_map = {
        'N': 'Normalni ritam',
        'A': 'Atrijalna fibrilacija',
        'O': 'Ostali ritmovi',
        '~': 'Šum'
    }
    rhythm_name = rhythm_map.get(label, label)
    
    # Kreiranje grafikona
    plt.figure(figsize=(12, 6))
    
    # Originalni signal
    plt.subplot(2, 1, 1)
    plt.plot(time, original_signal, 'b-', alpha=0.7)
    plt.title(f"Originalni signal ({rhythm_name})", fontsize=12)
    plt.ylabel("Amplituda", fontsize=10)
    plt.grid(alpha=0.3)
    
    # Normalizirani signal
    plt.subplot(2, 1, 2)
    plt.plot(time, normalized_signal, 'r-', alpha=0.7)
    plt.title(f"Normalizirani signal [0, 1]", fontsize=12)
    plt.xlabel("Vrijeme (sekunde)", fontsize=10)
    plt.ylabel("Normalizirana vrijednost", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_normalization()
    verify_normalization()