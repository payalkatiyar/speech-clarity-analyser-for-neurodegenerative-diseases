import librosa
import numpy as np
import os

def raw_clarity_score(label, session_name, audio_path):
    """
    Step 1: Raw clarity estimate (NOT normalized)
    """

    # ---- Base score by speaker type + session ----
    if label == "normal":
        base = 0.9
    else:
        session = session_name.lower()
        if "session1" in session:
            base = 0.75
        elif "session2" in session:
            base = 0.55
        elif "session3" in session:
            base = 0.35
        else:
            base = 0.30

    # ---- Utterance-level variation ----
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.sqrt(np.mean(y**2))

        duration_factor = np.clip(duration / 5.0, 0.8, 1.2)
        rms_factor = np.clip(rms / 0.05, 0.8, 1.2)

        base *= 0.6 * duration_factor + 0.4 * rms_factor
    except:
        pass

    return float(base)


def normalize_scores_within_speaker(scores):
    """
    Min-max normalize clarity scores for ONE speaker
    """
    scores = np.array(scores)

    min_s = scores.min()
    max_s = scores.max()

    if max_s - min_s < 1e-6:
        return np.ones_like(scores) * 0.5  # fallback

    return (scores - min_s) / (max_s - min_s)
