import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

def robust_r_peak_detection(signal, sampling_rate=150):
    """
    Robusna detekcija R-pikova optimizovana za AF signale
    """
    print(f"Signal dužina: {len(signal)}, sampling rate: {sampling_rate}")
    
    # Korak 1: Poboljšano preprocessing
    cleaned_signal = preprocess_signal(signal, sampling_rate)
    
    # Korak 2: Detekcija R-pikova sa više metoda
    r_peaks, detection_method = detect_r_peaks_multimethod(cleaned_signal, sampling_rate)
    
    # Korak 3: Striktno eliminisanje lažnih pikova za AF
    if len(r_peaks) >= 3:
        r_peaks = eliminate_false_peaks_af(r_peaks, cleaned_signal, sampling_rate)
    
    return r_peaks, detection_method, cleaned_signal

def preprocess_signal(signal, sampling_rate):
    """
    Poboljšano preprocessing sa bandpass filterom
    """
    try:
        # 1. Ukloni baseline drift sa high-pass filterom
        nyquist = sampling_rate / 2
        low_cutoff = 0.5 / nyquist
        high_cutoff = 40 / nyquist
        
        b, a = butter(3, [low_cutoff, high_cutoff], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        
        # 2. Blago smoothing
        cleaned_signal = savgol_filter(filtered_signal, window_length=5, polyorder=2)
        
        # 3. NeuroKit cleaning kao finalni korak
        cleaned_signal = nk.ecg_clean(cleaned_signal, sampling_rate=sampling_rate, method="neurokit")
        
        print("Preprocessing završen uspešno")
        return cleaned_signal
        
    except Exception as e:
        print(f"Greška u preprocessing: {e}")
        return signal.copy()

def detect_r_peaks_multimethod(signal, sampling_rate):
    """
    Probaj više metoda za detekciju R-pikova
    """
    methods = [
        ("pantompkins1985", lambda s: nk.ecg_peaks(s, sampling_rate=sampling_rate, method="pantompkins1985")),
        ("hamilton2002", lambda s: nk.ecg_peaks(s, sampling_rate=sampling_rate, method="hamilton2002")),
        ("christov2004", lambda s: nk.ecg_peaks(s, sampling_rate=sampling_rate, method="christov2004")),
        ("neurokit", lambda s: nk.ecg_process(s, sampling_rate=sampling_rate)[1])
    ]
    
    best_peaks = []
    best_method = "none"
    
    for method_name, method_func in methods:
        try:
            if method_name == "neurokit":
                info = method_func(signal)
            else:
                _, info = method_func(signal)
            
            r_peaks = info["ECG_R_Peaks"]
            
            if len(r_peaks) >= 5:  # Minimum 5 pikova za pouzdanu analizu
                print(f"Metoda {method_name}: {len(r_peaks)} pikova")
                best_peaks = r_peaks
                best_method = method_name
                break
                
        except Exception as e:
            print(f"Metoda {method_name} neuspešna: {e}")
            continue
    
    # Fallback na scipy ako ništa ne radi
    if len(best_peaks) < 5:
        try:
            # Adaptivni prag na osnovu signala
            signal_std = np.std(signal)
            signal_mean = np.mean(signal)
            threshold = signal_mean + 1.5 * signal_std
            
            r_peaks, _ = find_peaks(signal, 
                                  height=threshold,
                                  distance=int(0.25 * sampling_rate),  # Min 250ms između pikova
                                  prominence=signal_std * 0.5)
            
            best_peaks = r_peaks
            best_method = "scipy_adaptive"
            print(f"Scipy fallback: {len(best_peaks)} pikova, prag: {threshold:.3f}")
            
        except Exception as e:
            print(f"I scipy fallback neuspešan: {e}")
    
    return best_peaks, best_method

def eliminate_false_peaks_af(r_peaks, signal, sampling_rate):
    """
    Specifično eliminisanje lažnih pikova za AF signale - strožiji pristup
    """
    if len(r_peaks) < 3:
        return r_peaks
    
    print(f"    AF eliminacija - početak: {len(r_peaks)} R-pikova")
    
    # Korak 1: Ukloni edge artefakte
    r_peaks = remove_edge_artifacts(r_peaks, signal, sampling_rate)
    
    # Korak 2: Striktno filtriranje po RR intervalima za AF
    r_peaks = filter_rr_intervals_af(r_peaks, signal, sampling_rate)
    
    # Korak 3: Amplitude-based filtering
    r_peaks = filter_by_amplitude_strict(r_peaks, signal)
    
    # Korak 4: Refractory period check
    r_peaks = apply_refractory_period(r_peaks, sampling_rate)
    
    print(f"    AF eliminacija - kraj: {len(r_peaks)} R-pikova")
    
    return r_peaks

def filter_rr_intervals_af(r_peaks, signal, sampling_rate):
    """
    Strožije filtriranje RR intervala specifično za AF
    """
    if len(r_peaks) < 3:
        return r_peaks
    
    # Izračunaj RR intervale u ms
    rr_intervals = np.diff(r_peaks) / sampling_rate * 1000
    
    print(f"    RR opseg: {np.min(rr_intervals):.0f}-{np.max(rr_intervals):.0f}ms")
    print(f"    RR srednja vrednost: {np.mean(rr_intervals):.0f}ms, std: {np.std(rr_intervals):.0f}ms")
    
    # Za AF: očekujemo velike varijacije, ali i dalje fiziološke granice
    # Strože granice za eliminaciju očiglednih grešaka
    MIN_RR_AF = 200  # 300 bpm max
    MAX_RR_AF = 2500  # 24 bpm min
    
    # Dodatno: ukloni ekstremne outliere (više od 3 IQR)
    q25, q75 = np.percentile(rr_intervals, [25, 75])
    iqr = q75 - q25
    
    # Za AF signale, očekujemo veliki IQR, ali ne ekstremne vrednosti
    lower_bound = max(MIN_RR_AF, q25 - 3 * iqr)
    upper_bound = min(MAX_RR_AF, q75 + 3 * iqr)
    
    print(f"    Granice filtriranja: {lower_bound:.0f}-{upper_bound:.0f}ms")
    
    # Iterativno uklanjanje najgorih outliera
    valid_peaks = list(r_peaks)
    removed_count = 0
    
    for iteration in range(20):  # Max 20 iteracija
        if len(valid_peaks) < 3:
            break
            
        current_rr = np.diff(valid_peaks) / sampling_rate * 1000
        
        # Nađi najgori outlier
        worst_idx = None
        worst_deviation = 0
        
        for i, rr in enumerate(current_rr):
            if rr < lower_bound or rr > upper_bound:
                deviation = min(abs(rr - lower_bound), abs(rr - upper_bound))
                if deviation > worst_deviation:
                    worst_deviation = deviation
                    worst_idx = i
        
        if worst_idx is None:
            break
        
        # Ukloni pik koji pravi najgori RR interval
        # Biraš između i-tog i (i+1)-og pika na osnovu amplitude
        peak1_amp = signal[valid_peaks[worst_idx]]
        peak2_amp = signal[valid_peaks[worst_idx + 1]]
        
        if peak1_amp < peak2_amp:
            remove_idx = worst_idx
        else:
            remove_idx = worst_idx + 1
        
        removed_rr = current_rr[worst_idx]
        valid_peaks.pop(remove_idx)
        removed_count += 1
        
        print(f"    Iter {iteration+1}: uklonjen RR={removed_rr:.0f}ms")
    
    print(f"    Ukupno uklonjeno: {removed_count} pikova")
    return np.array(valid_peaks)

def filter_by_amplitude_strict(r_peaks, signal):
    """
    Strožije amplitude filtering
    """
    if len(r_peaks) < 3:
        return r_peaks
    
    amplitudes = signal[r_peaks]
    median_amp = np.median(amplitudes)
    mad_amp = np.median(np.abs(amplitudes - median_amp))
    
    # Strožiji prag - ukloni pikove sa značajno manjom amplitudom
    min_amplitude = median_amp - 2.5 * mad_amp  # Umesto 3, koristi 2.5
    
    valid_peaks = []
    removed_count = 0
    
    for peak in r_peaks:
        if signal[peak] >= min_amplitude:
            valid_peaks.append(peak)
        else:
            removed_count += 1
    
    if removed_count > 0:
        print(f"    Amplitude filter: uklonjeno {removed_count} pikova")
    
    return np.array(valid_peaks)

def apply_refractory_period(r_peaks, sampling_rate):
    """
    Primeni refractory period - minimalno vreme između R-pikova
    """
    if len(r_peaks) < 2:
        return r_peaks
    
    refractory_samples = int(0.2 * sampling_rate)  # 200ms refractory period
    
    valid_peaks = [r_peaks[0]]  # Prvi pik je uvek valjan
    
    for peak in r_peaks[1:]:
        if peak - valid_peaks[-1] >= refractory_samples:
            valid_peaks.append(peak)
    
    removed = len(r_peaks) - len(valid_peaks)
    if removed > 0:
        print(f"    Refractory period: uklonjeno {removed} pikova")
    
    return np.array(valid_peaks)

def remove_edge_artifacts(r_peaks, signal, sampling_rate):
    """Ukloni pikove blizu početka/kraja"""
    edge_samples = int(0.5 * sampling_rate)  # 0.5s sa oba kraja
    
    filtered_peaks = []
    for peak in r_peaks:
        if edge_samples <= peak <= len(signal) - edge_samples:
            filtered_peaks.append(peak)
    
    return np.array(filtered_peaks)

def analyze_rhythm_detailed(rr_intervals):
    """
    Detaljnija analiza ritma sa AF metrikama
    """
    if len(rr_intervals) < 10:
        return "Nedovoljno podataka"
    
    # Osnovne statistike
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    cv_rr = std_rr / mean_rr  # Coefficient of variation
    
    # AF-specifični parametri
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
    
    # pNN50 (procenat uzastopnih RR razlika > 50ms)
    nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
    pnn50 = nn50 / (len(rr_intervals) - 1) * 100
    
    # Irregularity index
    irregularity = np.mean(np.abs(np.diff(rr_intervals)))
    
    print(f"\n📊 Detaljna analiza ritma:")
    print(f"   Broj RR intervala: {len(rr_intervals)}")
    print(f"   Mean RR: {mean_rr:.1f}ms (HR: {60000/mean_rr:.1f} bpm)")
    print(f"   Std RR: {std_rr:.1f}ms (CV: {cv_rr:.3f})")
    print(f"   RMSSD: {rmssd:.1f}ms")
    print(f"   pNN50: {pnn50:.1f}%")
    print(f"   Irregularity index: {irregularity:.1f}ms")
    
    # AF klasifikacija (strožiji kriterijumi)
    af_score = 0
    
    if cv_rr > 0.12:  # Visoka varijabilnost
        af_score += 2
    elif cv_rr > 0.08:
        af_score += 1
    
    if rmssd > 40:  # Visoka kratkoročna varijabilnost
        af_score += 2
    elif rmssd > 25:
        af_score += 1
    
    if pnn50 > 15:  # Visok procenat velikih promena
        af_score += 1
    
    if irregularity > 30:  # Visoka nepravilnost
        af_score += 1
    
    # Klasifikacija
    if af_score >= 4:
        rhythm_type = "AF (visoka verovatnoća)"
    elif af_score >= 2:
        rhythm_type = "Nepravilan ritam (moguće AF)"
    elif cv_rr > 0.05:
        rhythm_type = "Blago nepravilan ritam"
    else:
        rhythm_type = "Regularan ritam"
    
    print(f"   AF score: {af_score}/6")
    print(f"   🔍 Procena: {rhythm_type}")
    
    return rhythm_type

def create_enhanced_poincare_plot(rr_intervals, filename="enhanced_poincare.png"):
    """
    Kreiranje poboljšanog Poincaré plota sa analizom
    """
    if len(rr_intervals) < 2:
        print("Nedovoljno RR intervala za Poincaré plot")
        return
    
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]
    
    # Poincaré parametri
    diff_rr = rr_n1 - rr_n
    sum_rr = rr_n1 + rr_n
    
    sd1 = np.std(diff_rr) / np.sqrt(2)  # Short-term variability
    sd2 = np.std(sum_rr) / np.sqrt(2)   # Long-term variability
    
    # Kreiranje plota
    plt.figure(figsize=(8, 8))
    plt.scatter(rr_n, rr_n1, c='red', s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Dodaj elipse SD1 i SD2 za bolju vizuelizaciju
    mean_rr = np.mean(rr_intervals)
    
    # Identitet linija
    min_rr = min(np.min(rr_n), np.min(rr_n1))
    max_rr = max(np.max(rr_n), np.max(rr_n1))
    plt.plot([min_rr, max_rr], [min_rr, max_rr], 'k--', alpha=0.5, linewidth=1)
    
    plt.xlabel('RR(n) [ms]', fontsize=12)
    plt.ylabel('RR(n+1) [ms]', fontsize=12)
    plt.title(f'Poincaré plot\nSD1={sd1:.1f}ms, SD2={sd2:.1f}ms', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(300,1200)
    plt.ylim(300,1200)
    # Prikaži plot
    plt.tight_layout()
    plt.show()
    
    print(f"\n📈 Poincaré parametri:")
    print(f"   SD1: {sd1:.1f}ms (kratkoročna varijabilnost)")
    print(f"   SD2: {sd2:.1f}ms (dugoročna varijabilnost)")
    print(f"   SD1/SD2 ratio: {sd1/sd2:.3f}")
    
    # Interpretacija
    if sd1 > 25 and sd2 > 50:
        interpretation = "Visoka varijabilnost - karakteristično za AF"
    elif sd1 > 15:
        interpretation = "Umereena kratkoročna varijabilnost"
    else:
        interpretation = "Niska varijabilnost - regularan ritam"
    
    print(f"   Interpretacija: {interpretation}")

# Glavna analiza
def main():
    filename = "af.csv"
    
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Fajl {filename} nije pronađen!")
        return
    
    # Nađi liniju za A08328_1
    target_lines = [line for line in lines if line.startswith("A08328_1")]
    if not target_lines:
        print("Signal A08328_1 nije pronađen!")
        return
    
    target_line = target_lines[0]
    signal_str = target_line.strip().split(",")[1:]
    ecg_signal = np.array([float(x) for x in signal_str])
    
    print("🔍 POBOLJŠANA AF ANALIZA ECG signala A08328_1")
    print("=" * 60)
    
    # Primeni poboljšanu detekciju
    r_peaks, method, cleaned_signal = robust_r_peak_detection(ecg_signal, sampling_rate=150)
    print(f"🔧 Korišćena metoda: {method}")
    
    if len(r_peaks) >= 5:
        # Izračunaj RR intervale
        rr_intervals = np.diff(r_peaks) / 150.0 * 1000
        
        print(f"\n📋 Osnovne statistike RR intervala:")
        print(f"   Broj intervala: {len(rr_intervals)}")
        print(f"   Opseg: {np.min(rr_intervals):.1f} - {np.max(rr_intervals):.1f}ms")
        print(f"   Mean ± Std: {np.mean(rr_intervals):.1f} ± {np.std(rr_intervals):.1f}ms")
        
        # Detaljnija analiza ritma
        rhythm_type = analyze_rhythm_detailed(rr_intervals)
        
        # Kreiranje Poincaré plota
        create_enhanced_poincare_plot(rr_intervals)
        
        # Prikaži ECG sa detektovanim pikovima
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(ecg_signal, 'b-', linewidth=0.8, label='Originalni ECG')
        plt.plot(cleaned_signal, 'g-', linewidth=0.8, alpha=0.7, label='Čišćeni ECG')
        plt.scatter(r_peaks, cleaned_signal[r_peaks], color='red', s=50, zorder=5, label=f'R-pikovi (n={len(r_peaks)})')
        plt.title('ECG signal sa detektovanim R-pikovima')
        plt.ylabel('Amplituda')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(rr_intervals, 'ro-', markersize=4, linewidth=1)
        plt.title('RR intervali')
        plt.xlabel('Redni broj')
        plt.ylabel('RR interval [ms]')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n✅ Analiza završena uspešno!")
        print(f"🏥 Finalna procena: {rhythm_type}")
        
    else:
        print("❌ Nedovoljno R-pikova za pouzdanu analizu!")
        print(f"   Detektovano samo {len(r_peaks)} pikova")
        
        # Prikaži šta je detektovano
        plt.figure(figsize=(15, 6))
        plt.plot(ecg_signal, 'b-', linewidth=0.8)
        if len(r_peaks) > 0:
            plt.scatter(r_peaks, ecg_signal[r_peaks], color='red', s=50)
        plt.title('ECG signal - neuspešna detekcija')
        plt.show()

if __name__ == "__main__":
    main()