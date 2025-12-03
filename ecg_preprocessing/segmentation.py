import h5py
import numpy as np

# Parametri
segment_duration_sec = 30
sampling_rate = 150
segment_length = segment_duration_sec * sampling_rate  # 4500
overlap = 0.5
step = int(segment_length * (1 - overlap))  # 2250

# Ulazni i izlazni fajl
INPUT_H5 = "normalized_ecg_signals_new.h5"
OUTPUT_H5 = "segmented_ecg_signals_30s.h5"

with h5py.File(INPUT_H5, 'r') as infile, h5py.File(OUTPUT_H5, 'w') as outfile:
    signals = infile['normalized_signals']
    labels = infile['labels']
    filenames = infile['filenames']
    
    # Priprema za izlazne vlen datasetove
    dt = h5py.vlen_dtype(np.float32)
    segmented_signals = outfile.create_dataset('segmented_signals', shape=(0,), maxshape=(None,), dtype=dt)
    segmented_labels = outfile.create_dataset('segmented_labels', shape=(0,), maxshape=(None,), dtype='S1')  # npr. b'N'
    segmented_filenames = outfile.create_dataset('segmented_filenames', shape=(0,), maxshape=(None,), dtype='S20')

    seg_count = 0
    for i in range(len(signals)):
        signal = np.array(signals[i])
        label = labels[i]
        fname = filenames[i].decode('utf-8')

        if len(signal) < segment_length:
            continue  # preskoči prekratke

        segment_num = 1
        for start in range(0, len(signal) - segment_length + 1, step):
            end = start + segment_length
            segment = signal[start:end]

            # Kreiraj novi naziv segmenta
            segment_name = f"{fname}_{segment_num}"

            # Proširi izlazne datasetove za novi unos
            segmented_signals.resize((seg_count + 1,))
            segmented_signals[seg_count] = segment.astype('float32')

            segmented_labels.resize((seg_count + 1,))
            segmented_labels[seg_count] = label

            segmented_filenames.resize((seg_count + 1,))
            segmented_filenames[seg_count] = segment_name.encode('utf-8')

            segment_num += 1
            seg_count += 1

    print(f"Završeno: napravljeno {seg_count} segmenata po 30s.")
