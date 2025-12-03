import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class AFibDetectorPDF:
    def __init__(self):
        self.scaler = StandardScaler()
        self.svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        self.cluster_threshold_lower = None
        self.cluster_threshold_upper = None
        
    def detect_qrs_peaks(self, ecg_signal, sampling_rate):
        """Poboljaana detekcija QRS kompleksa"""
        # Agresivnije filtriranje za AFib
        filtered_ecg = nk.signal_filter(ecg_signal, sampling_rate=sampling_rate, 
                                      lowcut=1.0, highcut=40)
        
        # Probaj razliite metode detekcije
        methods = ["pantompkins1985", "nabian2018", "hamilton2002"]
        best_rpeaks = []
        
        for method in methods:
            try:
                _, rpeaks_info = nk.ecg_peaks(
                    filtered_ecg,
                    sampling_rate=sampling_rate,
                    method=method,
                    correct_artifacts=True
                )
                rpeaks = rpeaks_info['ECG_R_Peaks']
                
                if len(rpeaks) > len(best_rpeaks):
                    best_rpeaks = rpeaks
                    
            except:
                continue
                
        return np.array(best_rpeaks)
    
    def calculate_rr_intervals(self, rpeaks, sampling_rate):
        """Izraun RR intervala sa poboljaanim iaenjem"""
        if len(rpeaks) < 2:
            return np.array([])
        
        rr_intervals = np.diff(rpeaks) / sampling_rate * 1000
        
        # Agresivnije filtriranje za AFib (airok raspon)
        rr_intervals = rr_intervals[(rr_intervals > 150) & (rr_intervals < 3000)]
        
        # Ukloni ekstremne outliere (viae od 3 std devijacije)
        if len(rr_intervals) > 3:
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            
            # Samo ukloni VRLO ekstremne vrijednosti
            rr_intervals = rr_intervals[
                (rr_intervals > mean_rr - 4*std_rr) & 
                (rr_intervals < mean_rr + 4*std_rr)
            ]
        
        return rr_intervals
    
    def create_poincare_plot(self, rr_intervals):
        """Kreiranje Poincaré plota"""
        if len(rr_intervals) < 2:
            return np.array([]), np.array([])
        
        x = rr_intervals[:-1]
        y = rr_intervals[1:]
        
        return x, y
    
    def count_clusters_density_based(self, x, y):
        """Brojanje klastera pomou DBSCAN (bolje za AFib)"""
        if len(x) < 3:
            return 1
        
        # Kreiranje matrice podataka
        points = np.column_stack([x, y])
        
        # Normalizacija za DBSCAN
        points_norm = (points - np.mean(points, axis=0)) / (np.std(points, axis=0) + 1e-8)
        
        # DBSCAN parametri - prilagoeni za Poincaré plotove
        eps_values = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        min_samples = max(2, len(points) // 20)  # Adaptivno
        
        best_n_clusters = 1
        best_score = -1
        
        for eps in eps_values:
            try:
                clustering = DBSCAN(eps=eps, min_samples=min_samples)
                labels = clustering.fit_predict(points_norm)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if n_clusters > 0:
                    # Score kombinira broj klastera i buku
                    score = n_clusters - (n_noise / len(points))
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                        
            except:
                continue
        
        # Posebno: provjeri je li uope klaster struktura
        # AFib esto ima "cloud" oblik - jedan veliki difuzan klaster
        spread = np.std(points, axis=0)
        total_spread = np.mean(spread)
        
        # Ako je spread velik i nema jasne klastere, oznaava AFib pattern
        if total_spread > np.mean(x) * 0.15:  # 15% od prosjenog RR
            # Ovo je vjerojatno AFib - difuzan cloud
            return max(best_n_clusters, 8)  # "Previae" klastera
        
        return best_n_clusters
    
    def calculate_mean_stepping_increment(self, rr_intervals):
        """PDF formula: prosjeni korak promjene"""
        if len(rr_intervals) < 3:
            return 0
        
        x, y = self.create_poincare_plot(rr_intervals)
        
        if len(x) < 2:
            return 0
        
        # Udaljenosti izmeu uzastopnih toaka (PDF formula)
        total_distance = 0
        for i in range(len(x) - 1):
            dist = np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
            total_distance += dist
        
        mean_stepping = total_distance / (len(x) - 1)
        
        # Normalizacija sa srednjim RR
        mean_rr = np.mean(rr_intervals)
        normalized_stepping = mean_stepping / mean_rr if mean_rr > 0 else 0
        
        return normalized_stepping
    
    def calculate_dispersion_pdf(self, rr_intervals):
        """PDF formula: raspraenost oko dijagonale"""
        if len(rr_intervals) < 2:
            return 0
        
        x, y = self.create_poincare_plot(rr_intervals)
        
        if len(x) == 0:
            return 0
        
        # PDF pristup: centralna toka na dijagonali
        # a = (1/(2n-2)) * sum(RR_i + RR_{i+1})
        n = len(rr_intervals)
        central_point = sum(rr_intervals[i] + rr_intervals[i+1] 
                           for i in range(n-1)) / (2*(n-1))
        
        # Udaljenosti od dijagonale
        distances = []
        for i in range(len(x)):
            # Distance from point (x[i], y[i]) to diagonal line
            dist_to_diagonal = abs(y[i] - x[i]) / np.sqrt(2)
            distances.append(dist_to_diagonal)
        
        if len(distances) > 1:
            # PDF formula za dispersion
            mean_dist = np.mean(distances)
            var_dist = np.var(distances, ddof=1)
            std_dist = np.sqrt(var_dist)
            
            # Normalizacija
            mean_rr = np.mean(rr_intervals)
            dispersion = std_dist / mean_rr if mean_rr > 0 else 0
        else:
            dispersion = 0
            
        return dispersion
    
    def extract_features(self, ecg_signal, sampling_rate):
        """Ekstraktiranje tri PDF znaajke"""
        # Detekcija QRS vrhova
        rpeaks = self.detect_qrs_peaks(ecg_signal, sampling_rate)
        
        if len(rpeaks) < 10:
            return None, None
        
        # RR intervali
        rr_intervals = self.calculate_rr_intervals(rpeaks, sampling_rate)
        
        if len(rr_intervals) < 8:
            return None, None
        
        # Poincaré koordinate
        x, y = self.create_poincare_plot(rr_intervals)
        
        # Tri PDF znaajke
        n_clusters = self.count_clusters_density_based(x, y)
        mean_stepping = self.calculate_mean_stepping_increment(rr_intervals)
        dispersion = self.calculate_dispersion_pdf(rr_intervals)
        
        # Dodatne dijagnostike znaajke
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        cv_rr = rr_std / rr_mean if rr_mean > 0 else 0
        
        # Poincaré SD1/SD2 za dodatnu validaciju
        diff_rr = np.diff(rr_intervals)
        sd1 = np.sqrt(np.var(diff_rr, ddof=1) / 2) if len(diff_rr) > 1 else 0
        sd2 = np.sqrt(2 * np.var(rr_intervals, ddof=1) - sd1**2) if len(rr_intervals) > 1 else 0
        sd1_sd2_ratio = sd1 / sd2 if sd2 > 0 else 0
        
        features = {
            'n_clusters': n_clusters,
            'mean_stepping': mean_stepping,
            'dispersion': dispersion,
            'rr_intervals': rr_intervals,
            'poincare_x': x,
            'poincare_y': y,
            'cv_rr': cv_rr,
            'sd1_sd2_ratio': sd1_sd2_ratio,
            'rr_std': rr_std,
            'rr_mean': rr_mean
        }
        
        return features, rpeaks
    
    def classify_rhythm_pdf(self, features):
        """Klasifikacija prema originalnom PDF algoritmu + poboljaanja"""
        if features is None:
            return "Unknown", 0.0
        
        n_clusters = features['n_clusters']
        mean_stepping = features['mean_stepping']
        dispersion = features['dispersion']
        cv_rr = features['cv_rr']
        sd1_sd2_ratio = features['sd1_sd2_ratio']
        
        # === PDF ALGORITAM ===
        
        # KORAK 1: Analiza broja klastera
        # PDF: "limited number of clusters" = non-AFib pattern
        # Ali: 1 klaster ili "previae" klastera = potencijalni AFib
        
        cluster_evidence = ""
        afib_evidence_score = 0
        
        if n_clusters == 1:
            # Jedan klaster mo~e biti:
            # - Normalni ritam (kompaktan klaster)
            # - AFib (difuzan cloud)
            cluster_evidence = "Jedan klaster - potrebna dodatna analiza"
            afib_evidence_score += 0.2  # Neutralno
            
        elif 2 <= n_clusters <= 4:
            # "Limited number" - tipino non-AFib sa PVC ili aritmijama
            cluster_evidence = f"{n_clusters} klastera - ukazuje na strukturirani pattern"
            afib_evidence_score -= 0.3  # Protiv AFib
            
        elif n_clusters >= 5:
            # "Too many" clusters - AFib karakteristika
            cluster_evidence = f"{n_clusters} klastera - previae za regularan ritam"
            afib_evidence_score += 0.4  # Za AFib
        
        # KORAK 2: Mean Stepping Increment
        # AFib: visok stepping (nepravilnost)
        # Non-AFib: nizak stepping (regularan)
        
        stepping_evidence = ""
        if mean_stepping > 0.15:  # Visok threshold za jasnu nepravilnost
            stepping_evidence = "Visok stepping increment - nepravilan ritam"
            afib_evidence_score += 0.4
        elif mean_stepping > 0.08:
            stepping_evidence = "Umjeren stepping increment - blaga nepravilnost"
            afib_evidence_score += 0.2
        else:
            stepping_evidence = "Nizak stepping increment - regularan ritam"
            afib_evidence_score -= 0.2
        
        # KORAK 3: Dispersion
        # AFib: visoka dispersion (airoko raspraene toke)
        
        dispersion_evidence = ""
        if dispersion > 0.08:  # Visok threshold
            dispersion_evidence = "Visoka dispersion - airoko raspraene toke"
            afib_evidence_score += 0.3
        elif dispersion > 0.04:
            dispersion_evidence = "Umjerena dispersion"
            afib_evidence_score += 0.1
        else:
            dispersion_evidence = "Niska dispersion - kompaktne toke"
            afib_evidence_score -= 0.1
        
        # KORAK 4: Dodatni kriteriji (poboljaanje PDF pristupa)
        
        # Koeficijent varijacije RR intervala (sna~an AFib indikator)
        cv_evidence = ""
        if cv_rr > 0.12:  # >12% varijacija je vrlo sumnjiva
            cv_evidence = "Vrlo visoka varijabilnost RR intervala"
            afib_evidence_score += 0.4
        elif cv_rr > 0.08:
            cv_evidence = "Visoka varijabilnost RR intervala"
            afib_evidence_score += 0.2
        
        # SD1/SD2 ratio (Poincaré elipsa)
        # AFib: visok SD1/SD2 (wide ellipse)
        poincare_evidence = ""
        if sd1_sd2_ratio > 0.5:
            poincare_evidence = "Visok SD1/SD2 ratio - nepravilnost kratkog roka"
            afib_evidence_score += 0.2
        
        # === FINALNA KLASIFIKACIJA ===
        
        # Prag odluke
        afib_threshold = 0.3
        
        if afib_evidence_score > afib_threshold:
            classification = "AFib"
            confidence = min(0.95, 0.5 + afib_evidence_score)
        else:
            classification = "Non-AFib"
            confidence = min(0.95, 0.5 - afib_evidence_score)
        
        # Detaljni ispis analize
        print(f"\n=== DETALJJNA ANALIZA PDF PRISTUPA ===")
        print(f"Broj klastera: {n_clusters} - {cluster_evidence}")
        print(f"Mean stepping: {mean_stepping:.6f} - {stepping_evidence}")
        print(f"Dispersion: {dispersion:.6f} - {dispersion_evidence}")
        print(f"CV RR: {cv_rr:.4f} - {cv_evidence}")
        print(f"SD1/SD2: {sd1_sd2_ratio:.4f} - {poincare_evidence}")
        print(f"AFib evidence score: {afib_evidence_score:.2f}")
        print(f"Threshold: {afib_threshold}")
        
        return classification, confidence
    
    def plot_comprehensive_analysis(self, features, classification, title_suffix=""):
        """Sveobuhvatan prikaz analize"""
        if features is None:
            return
        
        x = features['poincare_x']
        y = features['poincare_y']
        rr_intervals = features['rr_intervals']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Poincaré plot sa klasterima
        ax1 = axes[0, 0]
        scatter = ax1.scatter(x, y, alpha=0.7, s=30, c=range(len(x)), cmap='viridis')
        
        # Dodaj linije izmeu uzastopnih toaka (PDF pristup)
        for i in range(len(x)-1):
            ax1.plot([x[i], x[i+1]], [y[i], y[i+1]], 'gray', alpha=0.3, linewidth=0.5)
        
        # Dijagonalna linija
        min_val = min(np.min(x), np.min(y))
        max_val = max(np.max(x), np.max(y))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2)
        
        ax1.set_xlabel('RR(n) [ms]')
        ax1.set_ylabel('RR(n+1) [ms]')
        ax1.set_title(f'Poincaré Plot - {classification}')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. RR interval vremenska serija
        ax2 = axes[0, 1]
        ax2.plot(rr_intervals, 'b-', linewidth=1, marker='o', markersize=4, alpha=0.7)
        ax2.axhline(y=np.mean(rr_intervals), color='r', linestyle='--', alpha=0.7, label='Mean')
        ax2.fill_between(range(len(rr_intervals)), 
                        np.mean(rr_intervals) - np.std(rr_intervals),
                        np.mean(rr_intervals) + np.std(rr_intervals),
                        alpha=0.2, color='red', label='±1 SD')
        ax2.set_xlabel('Beat Number')
        ax2.set_ylabel('RR Interval [ms]')
        ax2.set_title('RR Interval Time Series')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribucija RR intervala
        ax3 = axes[0, 2]
        ax3.hist(rr_intervals, bins=20, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        ax3.axvline(x=np.mean(rr_intervals), color='r', linestyle='--', label='Mean')
        ax3.axvline(x=np.median(rr_intervals), color='g', linestyle='--', label='Median')
        ax3.set_xlabel('RR Interval [ms]')
        ax3.set_ylabel('Density')
        ax3.set_title('RR Interval Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. PDF znaajke
        ax4 = axes[1, 0]
        feature_names = ['Clusters', 'Mean\nStepping\n(×100)', 'Dispersion\n(×1000)']
        feature_values = [features['n_clusters'], 
                         features['mean_stepping']*100,
                         features['dispersion']*1000]
        
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        bars = ax4.bar(feature_names, feature_values, color=colors)
        ax4.set_title('PDF Features')
        ax4.set_ylabel('Scaled Values')
        
        # Dodaj vrijednosti na stupce
        original_values = [features['n_clusters'], features['mean_stepping'], features['dispersion']]
        for bar, val in zip(bars, original_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 5. Diferecije RR intervala (za SD1)
        ax5 = axes[1, 1]
        diff_rr = np.diff(rr_intervals)
        ax5.plot(diff_rr, 'g-', linewidth=1, marker='s', markersize=3, alpha=0.7)
        ax5.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax5.set_xlabel('Beat Number')
        ax5.set_ylabel('RR(n+1) - RR(n) [ms]')
        ax5.set_title('RR Interval Differences')
        ax5.grid(True, alpha=0.3)
        
        # 6. Sa~etak svih mjerenja
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = f"""
SA}ETAK ANALIZE:
        
PDF Znaajke:
" Broj klastera: {features['n_clusters']}
" Mean stepping: {features['mean_stepping']:.6f}
" Dispersion: {features['dispersion']:.6f}

Dodatne znaajke:
" CV RR: {features['cv_rr']:.4f}
" SD1/SD2: {features['sd1_sd2_ratio']:.4f}
" RR std: {features['rr_std']:.2f} ms
" RR mean: {features['rr_mean']:.2f} ms

Klasifikacija: {classification}
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle(f'Comprehensive AFib Analysis - {classification} {title_suffix}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# Glavna funkcija za analizu
def analyze_ecg_afib(ecg_data, sampling_rate=150):
    """Analiza ECG-a za AFib koristei poboljaani PDF pristup"""
    detector = AFibDetectorPDF()
    
    # Ekstraktiranje znaajki
    features, rpeaks = detector.extract_features(ecg_data, sampling_rate)
    
    if features is None:
        print("GRE`KA: Neuspjeana analiza - premalo QRS kompleksa")
        return None
    
    # Klasifikacija
    classification, confidence = detector.classify_rhythm_pdf(features)
    
    # Osnovni ispis
    print(f"\n=== POBOLJ`ANA PDF ANALIZA ===")
    print(f"QRS kompleksi: {len(rpeaks)}")
    print(f"RR intervali: {len(features['rr_intervals'])}")
    print(f"KLASIFIKACIJA: {classification} (pouzdanost: {confidence:.2f})")
    
    # Vizualizacija
    detector.plot_comprehensive_analysis(features, classification)
    
    return features, classification, confidence

# Pokretanje analize
if __name__ == "__main__":
    try:
        # Vaai podaci
        df = pd.read_csv('af.csv', header=None, low_memory=False)
        signal_row = df[df[0] == 'A08328_1'].iloc[0, 1:].dropna().values
        ecg_signal = signal_row.astype(float)
        
        print("Analiziram ECG signal...")
        result = analyze_ecg_afib(ecg_signal, sampling_rate=150)
        
    except Exception as e:
        print(f"Greaka: {e}")

        