import numpy as np
import librosa

from mlservice.utils.audio_processing import preprocess_audio, extract_clarity_features


def raw_clarity_score(label, session, audio_path, signal=None, sr=None):
    """
    Compute a clarity score that is ~70% audio-derived, ~30% prior.
    Returns a float typically in [0.0, 1.2] range (clipped later by normalization).
    """

    # ---------- Prior component (30%) ----------
    base = 0.85 if label == "normal" else 0.50

    session_penalty = {
        "session1": 0.0,
        "session2": -0.10,
        "session3": -0.20
    }.get(session.lower(), 0.0)

    prior_score = base + session_penalty  # range ~[0.30, 0.85]

    # ---------- Audio-derived component (70%) ----------
    try:
        if signal is None or sr is None:
            result = preprocess_audio(audio_path)
            if result is None:
                # Fallback: use only prior
                return float(np.clip(prior_score, 0.0, 1.0))
            signal, sr = result
        
        feats = extract_clarity_features(signal, sr)
    except Exception:
        return float(np.clip(prior_score, 0.0, 1.0))

    # Individual sub-scores, each normalized to ~[0, 1]

    # 1. HNR proxy: higher = clearer speech. Typical range [0, 1]
    hnr_score = np.clip(feats.get("hnr_proxy", 0.5), 0, 1)

    # 2. Voiced ratio: higher = more voiced frames = clearer
    voiced_score = np.clip(feats["voiced_ratio"], 0, 1)

    # 3. F0 stability: lower std relative to mean = more stable = clearer
    if feats["f0_mean"] > 0:
        f0_cv = feats["f0_std"] / (feats["f0_mean"] + 1e-6)
        f0_score = np.clip(1.0 - f0_cv, 0, 1)
    else:
        f0_score = 0.3  # no pitch detected = likely impaired

    # 4. RMS energy consistency: lower std = more stable energy = clearer
    if feats["rms_mean"] > 0:
        rms_cv = feats["rms_std"] / (feats["rms_mean"] + 1e-6)
        rms_score = np.clip(1.0 - rms_cv * 2, 0, 1)
    else:
        rms_score = 0.3

    # 5. Spectral flatness: lower = more tonal/harmonic = clearer speech
    flatness_score = np.clip(1.0 - feats["flatness_mean"] * 10, 0, 1)

    # 6. Duration: very short utterances are suspicious
    dur_score = np.clip(feats["duration"] / 3.0, 0, 1)

    # 7. Spectral centroid: moderate range is normal speech (~1000-3000 Hz)
    centroid_norm = feats["centroid_mean"] / (8000 / 2)  # normalize by Nyquist/2
    centroid_score = np.clip(1.0 - abs(centroid_norm - 0.4) * 2, 0, 1)

    # Weighted audio score
    audio_score = (
        0.25 * hnr_score +
        0.20 * voiced_score +
        0.15 * f0_score +
        0.15 * rms_score +
        0.10 * flatness_score +
        0.08 * dur_score +
        0.07 * centroid_score
    )

    # Combine: 30% prior + 70% audio
    combined = 0.30 * prior_score + 0.70 * audio_score

    return float(combined)


def normalize_scores_within_speaker(scores):
    """
    Min-max normalization within speaker.
    Fixed edge cases: constant scores → 0.5, not 0.0.
    """
    scores = np.array(scores, dtype=np.float64)

    if len(scores) < 2:
        # Single sample: use raw score clipped to [0, 1]
        return np.clip(scores, 0, 1)

    min_s = np.min(scores)
    max_s = np.max(scores)

    if max_s - min_s < 1e-8:
        # All scores identical → neutral value 0.5
        return np.full_like(scores, 0.5)

    norm = (scores - min_s) / (max_s - min_s)
    return np.clip(norm, 0, 1)