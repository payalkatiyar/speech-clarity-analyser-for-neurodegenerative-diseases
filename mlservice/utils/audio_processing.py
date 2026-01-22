import os
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

# ================= CONFIG =================
SAMPLE_RATE = 16000
MIN_DURATION = 0.5        # seconds
MIN_RMS = 0.005           # silence threshold
TOP_DB = 25               # silence trimming aggressiveness
N_MFCC = 40
MAX_LEN = 200             # MFCC time frames

# ================= BANDPASS FILTER =================
def bandpass_filter(signal, sr, low=80, high=8000):
    """
    Speech-safe bandpass filter with Nyquist protection
    """
    nyq = 0.5 * sr

    # Safety clamp
    low = max(low, 20)
    high = min(high, nyq * 0.99)

    if low >= high:
        return signal  # skip filtering safely

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
    Loads and conditions audio in memory.
    Deletes file ONLY if delete_if_silent=True and audio is invalid.
    """

    try:
        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"❌ Error loading {audio_path}: {e}")
        if delete_if_silent and os.path.exists(audio_path):
            os.remove(audio_path)
        return None

    # Trim leading & trailing silence
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

    # Band-pass filtering
    signal = bandpass_filter(signal, sr)

    # Loudness normalization
    signal = librosa.util.normalize(signal)

    return signal, sr

# ================= FEATURE EXTRACTION =================
def extract_mfcc(signal, sr):
    """
    Extracts MFCC features and pads/truncates to fixed length
    """
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Pad or truncate time dimension
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc

# ================= SAVE FILTERED AUDIO (LISTENING ONLY) =================
def save_filtered_audio(
    input_path,
    output_root="data/audio_filtered",
    preserve_structure=True
):
    """
    Saves a filtered copy of audio for listening/debugging ONLY.
    Does NOT overwrite original dataset.
    """

    result = preprocess_audio(input_path, delete_if_silent=False)
    if result is None:
        print("⚠️ Audio invalid or too silent. Not saved.")
        return None

    signal, sr = result

    if preserve_structure:
        # Preserve folder structure from data/audio
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
