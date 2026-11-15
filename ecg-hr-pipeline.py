"""
ECG Signal Processing and Heart Rate Estimation Benchmarking Pipeline
Description:
    Automated discrete-time ECG preprocessing, R-peak detection, RR-interval 
    extraction, conventional time-domain heart rate calculation, and record-grouped 
    Random Forest regression benchmarking using the PhysioNet MIT-BIH Arrhythmia Database.
"""

import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================================================================
# 1. CONFIGURATION & EXPERIMENTAL PARAMETERS
# ==============================================================================
RECORDS = ['100', '101', '103', '105', '106', '115', '119', '200', '201', '213']
FS = 360.0                  # Nominal sampling frequency (Hz)
LOWCUT = 5.0                # Bandpass low cutoff (Hz)
HIGHCUT = 30.0              # Bandpass high cutoff (Hz)
FILTER_ORDER = 4            # Butterworth filter order
TOLERANCE_MS = 50.0         # R-peak matching tolerance (+/- 50 ms)
WINDOW_SEC = 10.0           # Analysis window length in seconds
TEST_RECORDS = ['200', '201', '213']  # Held-out unseen subjects for record-grouped split

# ==============================================================================
# 2. SIGNAL PREPROCESSING (BUTTERWORTH BANDPASS FILTER)
# ==============================================================================
def butter_bandpass_filter(signal, lowcut=LOWCUT, highcut=HIGHCUT, fs=FS, order=FILTER_ORDER):
    """Applies a zero-phase digital Butterworth bandpass filter to attenuate baseline wander and EMG noise."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# ==============================================================================
# 3. ALGORITHMIC R-PEAK DETECTION & EVALUATION
# ==============================================================================
def detect_r_peaks(filtered_ecg, fs=FS):
    """Detects fiducial R-peak sample indices using adaptive signal squaring and refractory thresholding."""
    # Derivative & non-linear squaring to accentuate QRS slopes
    diff_sig = np.diff(filtered_ecg, prepend=filtered_ecg[0])
    squared_sig = diff_sig ** 2
    
    # Moving-window integration (~150 ms)
    win_size = int(0.150 * fs)
    integrated_sig = np.convolve(squared_sig, np.ones(win_size) / win_size, mode='same')
    
    # Minimum refractory distance of 200 ms between R-peaks
    min_distance = int(0.200 * fs)
    threshold = 0.5 * np.mean(integrated_sig) + 0.5 * np.std(integrated_sig)
    
    candidate_peaks, _ = find_peaks(integrated_sig, distance=min_distance, height=threshold)
    
    # Refine peak locations to exact local maxima on the filtered ECG waveform
    refined_peaks = []
    search_radius = int(0.050 * fs)
    for p in candidate_peaks:
        start = max(0, p - search_radius)
        end = min(len(filtered_ecg), p + search_radius)
        local_max = start + np.argmax(filtered_ecg[start:end])
        refined_peaks.append(local_max)
        
    return np.unique(refined_peaks)

def evaluate_r_peaks(ref_peaks, det_peaks, fs=FS, tolerance_ms=TOLERANCE_MS):
    """Matches detected R-peaks against ground-truth annotations within +/- tolerance_ms (one-to-one)."""
    tol_samples = int((tolerance_ms / 1000.0) * fs)
    matched_ref = set()
    tp = 0
    
    for det in det_peaks:
        diffs = np.abs(ref_peaks - det)
        valid_indices = np.where(diffs <= tol_samples)[0]
        
        # Match to closest unmatched reference peak
        unmatched = [idx for idx in valid_indices if idx not in matched_ref]
        if unmatched:
            best_idx = unmatched[np.argmin(diffs[unmatched])]
            matched_ref.add(best_idx)
            tp += 1
            
    fp = len(det_peaks) - tp
    fn = len(ref_peaks) - tp
    return tp, fp, fn

# ==============================================================================
# 4. WINDOWED FEATURE EXTRACTION & HEART RATE ESTIMATION
# ==============================================================================
def extract_window_features(record_id, filtered_ecg, det_peaks, ref_peaks, fs=FS, window_sec=WINDOW_SEC):
    """Extracts statistical RR features and computes conventional vs. reference HR across time windows."""
    total_samples = len(filtered_ecg)
    win_samples = int(window_sec * fs)
    rows = []
    
    for start in range(0, total_samples - win_samples + 1, win_samples):
        end = start + win_samples
        
        w_det = det_peaks[(det_peaks >= start) & (det_peaks < end)]
        w_ref = ref_peaks[(ref_peaks >= start) & (ref_peaks < end)]
        
        # Require at least 2 R-peaks to form valid RR intervals
        if len(w_det) >= 2 and len(w_ref) >= 2:
            det_rr = np.diff(w_det) / fs
            ref_rr = np.diff(w_ref) / fs
            
            conv_hr = 60.0 / np.mean(det_rr)
            ref_hr = 60.0 / np.mean(ref_rr)
            
            rows.append({
                'record': record_id,
                'mean_rr': np.mean(det_rr),
                'median_rr': np.median(det_rr),
                'sd_rr': np.std(det_rr),
                'min_rr': np.min(det_rr),
                'max_rr': np.max(det_rr),
                'beat_count': len(w_det),
                'conv_hr': conv_hr,
                'ref_hr': ref_hr
            })
            
    return pd.DataFrame(rows)

# ==============================================================================
# 5. VISUALIZATION HELPERS
# ==============================================================================
def save_diagnostic_plots(raw_100, filt_100, det_100, ref_100, test_df, rf_preds, conv_metrics, rf_metrics):
    """Generates and saves evaluation plots into an output/ directory."""
    os.makedirs('output', exist_ok=True)
    t = np.arange(int(10 * FS)) / FS
    
    # Plot 1: Raw vs Filtered ECG (First 10s of Record 100)
    plt.figure(figsize=(10, 4))
    plt.plot(t, raw_100[:len(t)], label='Raw ECG', alpha=0.6)
    plt.plot(t, filt_100[:len(t)], label='Filtered ECG (5–30 Hz)', linewidth=1.2)
    plt.title('Raw and Filtered ECG Signal (Record 100)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/raw_vs_filtered_ecg.png', dpi=300)
    plt.close()
    
    # Plot 2: R-Peak Detection Overlay
    det_10s = det_100[det_100 < len(t)]
    ref_10s = ref_100[ref_100 < len(t)]
    plt.figure(figsize=(10, 4))
    plt.plot(t, filt_100[:len(t)], label='Filtered ECG')
    plt.plot(det_10s / FS, filt_100[det_10s], 'rx', markersize=8, label='Detected R-peaks')
    plt.plot(ref_10s / FS, filt_100[ref_10s], 'go', fillstyle='none', markersize=10, label='Reference Beats')
    plt.title('R-Peak Detection Verification (Record 100)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/r_peak_detection.png', dpi=300)
    plt.close()
    
    # Plot 3: Error Comparison Bar Chart
    labels = ['Conventional DSP', 'Random Forest']
    maes = [conv_metrics['MAE'], rf_metrics['MAE']]
    rmses = [conv_metrics['RMSE'], rf_metrics['RMSE']]
    
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, maes, width, label='MAE (bpm)')
    plt.bar(x + width/2, rmses, width, label='RMSE (bpm)')
    plt.xticks(x, labels)
    plt.ylabel('Error (bpm)')
    plt.title('Heart Rate Estimation Error Comparison')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/hr_error_comparison.png', dpi=300)
    plt.close()

# ==============================================================================
# 6. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    print("==================================================================")
    print(" ECG SIGNAL PROCESSING & HEART RATE ESTIMATION BENCHMARK PIPELINE")
    print("==================================================================\n")
    
    summary_rows = []
    all_windows = []
    total_tp, total_fp, total_fn = 0, 0, 0
    rec100_cache = {}
    
    # Valid beat annotation symbols in MIT-BIH
    beat_symbols = {'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'}
    
    for rec in RECORDS:
        print(f"Downloading & processing MIT-BIH Record {rec}...")
        record = wfdb.rdrecord(rec, pn_dir='mitdb')
        annotation = wfdb.rdann(rec, 'atr', pn_dir='mitdb')
        
        raw_ecg = record.p_signal[:, 0]
        ref_peaks = np.array([s for s, sym in zip(annotation.sample, annotation.symbol) if sym in beat_symbols])
        
        # 1. Bandpass Filtering
        filtered_ecg = butter_bandpass_filter(raw_ecg)
        
        # 2. R-Peak Detection & Evaluation
        det_peaks = detect_r_peaks(filtered_ecg)
        tp, fp, fn = evaluate_r_peaks(ref_peaks, det_peaks)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # 3. Windowed Feature & HR Extraction
        win_df = extract_window_features(rec, filtered_ecg, det_peaks, ref_peaks)
        all_windows.append(win_df)
        
        summary_rows.append({
            'Record': rec,
            'Reference Beats': len(ref_peaks),
            'Detected Beats': len(det_peaks),
            'Valid HR Windows': len(win_df)
        })
        
        if rec == '100':
            rec100_cache = {'raw': raw_ecg, 'filt': filtered_ecg, 'det': det_peaks, 'ref': ref_peaks}
            
    # Display Per-Record Summary Table
    summary_df = pd.DataFrame(summary_rows)
    print("\n--- Per-Record R-Peak Detection & Window Summary ---")
    print(summary_df.to_string(index=False))
    
    # Overall R-Peak Detection Performance
    sensitivity = (total_tp / (total_tp + total_fn)) * 100.0
    ppv = (total_tp / (total_tp + total_fp)) * 100.0
    f1_score = (2 * sensitivity * ppv) / (sensitivity + ppv)
    
    print("\n--- Overall R-Peak Detection Metrics (+/- 50 ms tolerance) ---")
    print(f"True Positives (TP) : {total_tp}")
    print(f"False Positives (FP): {total_fp}")
    print(f"False Negatives (FN): {total_fn}")
    print(f"Sensitivity         : {sensitivity:.4f}%")
    print(f"PPV                 : {ppv:.4f}%")
    print(f"F1-Score            : {f1_score:.4f}%")
    
    # Combine all extracted time windows
    full_df = pd.concat(all_windows, ignore_index=True)
    
    # Record-Grouped Train/Test Split (Prevents Cross-Subject Data Leakage)
    train_df = full_df[~full_df['record'].isin(TEST_RECORDS)].copy()
    test_df = full_df[full_df['record'].isin(TEST_RECORDS)].copy()
    
    feature_cols = ['mean_rr', 'median_rr', 'sd_rr', 'min_rr', 'max_rr', 'beat_count']
    X_train, y_train = train_df[feature_cols], train_df['ref_hr']
    X_test, y_test = test_df[feature_cols], test_df['ref_hr']
    
    # Train Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    # Evaluate Conventional DSP vs. Random Forest on Unseen Test Windows
    conv_preds = test_df['conv_hr'].values
    
    conv_metrics = {
        'MAE': mean_absolute_error(y_test, conv_preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, conv_preds)),
        'MBE': np.mean(conv_preds - y_test)
    }
    
    rf_metrics = {
        'MAE': mean_absolute_error(y_test, rf_preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_preds)),
        'MBE': np.mean(rf_preds - y_test)
    }
    
    print("\n--- Heart Rate Estimation Benchmark (Unseen Test Recordings) ---")
    comp_df = pd.DataFrame([
        {'Method': 'Conventional Time-Domain DSP', **conv_metrics},
        {'Method': 'Random Forest Regressor', **rf_metrics}
    ])
    print(comp_df.to_string(index=False))
    
    # Save diagnostic visualizations
    if rec100_cache:
        save_diagnostic_plots(
            rec100_cache['raw'], rec100_cache['filt'],
            rec100_cache['det'], rec100_cache['ref'],
            test_df, rf_preds, conv_metrics, rf_metrics
        )
        print("\nDiagnostic plots saved to ./output/")

if __name__ == "__main__":
    main()