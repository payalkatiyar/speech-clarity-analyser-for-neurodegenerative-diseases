import os
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

# ================= CONFIG =================
SAMPLE_RATE = 16000

MIN_DURATION = 0.5        # seconds (keep short pathological utterances)
MIN_RMS = 0.005           # silence threshold

TOP_DB = 35               # softer trimming for dysarthria
N_MFCC = 40               # base MFCC count
MAX_LEN = 200             # time frames after padding

# Final feature dimension = 40 MFCC + Δ + ΔΔ = 120


# ================= BANDPASS FILTER =================
def bandpass_filter(signal, sr, low=50, high=7500):
    """
    Speech-aware bandpass filter.
    Keeps breathiness, tremor, and hoarseness.
    """
    nyq = 0.5 * sr

    low = max(low, 20)
    high = min(high, nyq * 0.99)

    if low >= high:
        return signal

    b, a = butter(
        4,
        [low / nyq, high / nyq],
        btype="band"
    )
    return lfilter(b, a, signal)


# ================= SILENCE CHECK =================
def is_too_silent(signal):
    rms = np.sqrt(np.mean(signal ** 2))
    return rms < MIN_RMS


# ================= PREPROCESS AUDIO =================
def preprocess_audio(audio_path, delete_if_silent=False):
    """
    Loads and lightly conditions audio.
    DOES NOT over-clean (important for medical speech).
    """

    try:
        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"❌ Error loading {audio_path}: {e}")
        if delete_if_silent and os.path.exists(audio_path):
            os.remove(audio_path)
        return None

    # Trim leading & trailing silence (gentle)
    signal, _ = librosa.effects.trim(signal, top_db=TOP_DB)

    # Duration check
    duration = librosa.get_duration(y=signal, sr=sr)
    if duration < MIN_DURATION:
        if delete_if_silent and os.path.exists(audio_path):
            os.remove(audio_path)
        return None

    # RMS silence check
    if is_too_silent(signal):
        if delete_if_silent and os.path.exists(audio_path):
            os.remove(audio_path)
        return None

    # Speech-aware band-pass
    signal = bandpass_filter(signal, sr)

    # Loudness normalization (file-level)
    signal = librosa.util.normalize(signal)

    return signal, sr


# ================= FEATURE EXTRACTION =================
def extract_mfcc(signal, sr):
    """
    Extracts MFCC + Δ + ΔΔ features.
    Final shape: (120, MAX_LEN)
    """

    # Base MFCCs
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Temporal derivatives (CRITICAL for dysarthria)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack features
    features = np.vstack([mfcc, delta, delta2])  # (120, time)

    # Pad or truncate time dimension
    if features.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - features.shape[1]
        features = np.pad(
            features,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )
    else:
        features = features[:, :MAX_LEN]

    return features


# ================= CLARITY FEATURE EXTRACTION =================
def extract_clarity_features(signal, sr):
    """
    Extracts audio-level features that correlate with speech clarity.
    Returns a dict of feature values used for label generation.
    """

    features = {}

    # --- Zero Crossing Rate ---
    zcr = librosa.feature.zero_crossing_rate(signal)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # --- Spectral Centroid (brightness indicator) ---
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    features["centroid_mean"] = float(np.mean(centroid))
    features["centroid_std"] = float(np.std(centroid))

    # --- Spectral Rolloff ---
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    features["rolloff_mean"] = float(np.mean(rolloff))

    # --- RMS Energy ---
    rms = librosa.feature.rms(y=signal)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # --- Spectral Bandwidth ---
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    features["bandwidth_mean"] = float(np.mean(bandwidth))

    # --- Spectral Flatness (noise-like vs tonal) ---
    flatness = librosa.feature.spectral_flatness(y=signal)
    features["flatness_mean"] = float(np.mean(flatness))

    # --- HNR approximation (simpler/faster without HPSS) ---
    # use spectral flatness as a proxy for tonality/noise ratio
    features["hnr_proxy"] = float(1.0 - np.mean(flatness))

    # --- Duration ---
    features["duration"] = float(librosa.get_duration(y=signal, sr=sr))

    # --- Pitch stability (fast autocorrelation) ---
    try:
        # Use autocorrelation for fast pitch estimation
        frame_length = int(0.03 * sr)  # 30ms frames
        hop = int(0.01 * sr)  # 10ms hop
        f0_values = []

        for start in range(0, len(signal) - frame_length, hop):
            frame = signal[start:start + frame_length]
            if np.max(np.abs(frame)) < 0.01:
                continue
            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            # Find first peak after initial decline
            min_lag = int(sr / 500)  # max 500 Hz
            max_lag = int(sr / 50)   # min 50 Hz
            if max_lag > len(corr):
                continue
            segment = corr[min_lag:max_lag]
            if len(segment) > 0 and np.max(segment) > 0.3 * corr[0]:
                peak_idx = np.argmax(segment) + min_lag
                f0_values.append(sr / peak_idx)

        if len(f0_values) > 2:
            f0_arr = np.array(f0_values)
            features["f0_mean"] = float(np.mean(f0_arr))
            features["f0_std"] = float(np.std(f0_arr))
            features["voiced_ratio"] = float(len(f0_values) / max(1, len(signal) // hop))
        else:
            features["f0_mean"] = 0.0
            features["f0_std"] = 0.0
            features["voiced_ratio"] = 0.0
    except Exception:
        features["f0_mean"] = 0.0
        features["f0_std"] = 0.0
        features["voiced_ratio"] = 0.0

    return features


# ================= SAVE FILTERED AUDIO (DEBUG ONLY) =================
def save_filtered_audio(
    input_path,
    output_root="data/audio_filtered",
    preserve_structure=True
):
    """
    Saves a filtered copy for listening/debugging.
    Does NOT overwrite training data.
    """

    result = preprocess_audio(input_path, delete_if_silent=False)
    if result is None:
        print("⚠️ Audio invalid or too silent. Not saved.")
        return None

    signal, sr = result

    if preserve_structure:
        rel_path = input_path.replace("data/audio/", "")
        output_path = os.path.join(output_root, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        os.makedirs(output_root, exist_ok=True)
        output_path = os.path.join(
            output_root,
            os.path.basename(input_path)
        )

    sf.write(output_path, signal, sr)
    print(f"✅ Filtered audio saved at: {output_path}")
    return output_path
