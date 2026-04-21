import os
import json
import hashlib
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

from mlservice.utils.audio_processing import preprocess_audio, extract_mfcc
from mlservice.clarity_labels import raw_clarity_score, normalize_scores_within_speaker


CACHE_ROOT = "data/.cache"
LABEL_CACHE_DIR = os.path.join(CACHE_ROOT, "labels")
MFCC_CACHE_DIR = os.path.join(CACHE_ROOT, "mfcc")


class SpeechDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.samples = []
        self.augment = augment
        
        os.makedirs(LABEL_CACHE_DIR, exist_ok=True)
        os.makedirs(MFCC_CACHE_DIR, exist_ok=True)

        # Try loading from cache
        label_cache_path = self._label_cache_path(root_dir)
        if os.path.exists(label_cache_path):
            print(f"📦 Loading cached labels from {label_cache_path}")
            with open(label_cache_path, "r") as f:
                self.samples = [(s["path"], s["score"]) for s in json.load(f)]
        else:
            # -------- SINGLE PASS COLLECTION & LABELING --------
            print(f"🔍 First load: calculating labels for {root_dir}")
            speaker_items = defaultdict(list)
            
            # 1. Collect and preprocess (one-time load)
            for label in ["normal", "dysarthric"]:
                label_dir = os.path.join(root_dir, label)
                if not os.path.isdir(label_dir):
                    continue

                for mic in os.listdir(label_dir):
                    mic_dir = os.path.join(label_dir, mic)
                    if not os.path.isdir(mic_dir):
                        continue

                    for file in os.listdir(mic_dir):
                        if not file.endswith(".wav"):
                            continue

                        path = os.path.join(mic_dir, file)
                        result = preprocess_audio(path)
                        if result is None:
                            continue
                        
                        signal, sr = result
                        speaker = file.split("_")[0]
                        session = "session1"
                        if "session2" in file.lower(): session = "session2"
                        elif "session3" in file.lower(): session = "session3"

                        # Compute raw score immediately using pre-loaded signal
                        raw_score = raw_clarity_score(label, session, path, signal, sr)
                        
                        speaker_items[speaker].append({
                            "path": path,
                            "raw_score": raw_score
                        })

                        if (sum(len(v) for v in speaker_items.values())) % 1000 == 0:
                            print(f"  ... processed {sum(len(v) for v in speaker_items.values())} files")

            # 2. Normalize and finalize
            all_raw = []
            for speaker, items in speaker_items.items():
                all_raw.extend([it["raw_score"] for it in items])
            
            g_min = np.min(all_raw) if all_raw else 0.0
            g_max = np.max(all_raw) if all_raw else 1.0

            for speaker, items in speaker_items.items():
                raw_scores = [it["raw_score"] for it in items]
                if len(items) >= 3:
                    norm_scores = normalize_scores_within_speaker(raw_scores)
                else:
                    norm_scores = np.clip((np.array(raw_scores) - g_min) / (g_max - g_min + 1e-8), 0, 1)

                for it, score in zip(items, norm_scores):
                    self.samples.append((it["path"], float(score)))

            self._save_label_cache(label_cache_path)

        print(f"✅ Loaded {len(self.samples)} samples (augment={self.augment})")

    def _label_cache_path(self, root_dir):
        key = hashlib.md5(os.path.abspath(root_dir).encode()).hexdigest()
        return os.path.join(LABEL_CACHE_DIR, f"labels_{key}.json")

    def _save_label_cache(self, cache_path):
        data = [{"path": p, "score": s} for p, s in self.samples]
        with open(cache_path, "w") as f:
            json.dump(data, f)
        print(f"💾 Cached labels to {cache_path}")

    def _get_mfcc_cache_path(self, audio_path):
        key = hashlib.md5(audio_path.encode()).hexdigest()
        return os.path.join(MFCC_CACHE_DIR, f"{key}.npy")

    def __len__(self):
        return len(self.samples)

    def _spec_augment(self, mfcc):
        """Enhanced SpecAugment with time warping and stronger masking."""
        feat = mfcc.copy()
        n_freq, n_time = feat.shape

        # Frequency masking (up to 2 masks)
        num_freq_masks = np.random.randint(1, 3)
        for _ in range(num_freq_masks):
            if np.random.random() < 0.6:
                f_width = np.random.randint(1, min(12, n_freq // 4))
                f_start = np.random.randint(0, n_freq - f_width)
                feat[f_start:f_start + f_width, :] = 0.0

        # Time masking (up to 2 masks)
        num_time_masks = np.random.randint(1, 3)
        for _ in range(num_time_masks):
            if np.random.random() < 0.6:
                t_width = np.random.randint(1, min(25, n_time // 4))
                t_start = np.random.randint(0, n_time - t_width)
                feat[:, t_start:t_start + t_width] = 0.0

        # Time warping (simple stretch/compress via interpolation)
        if np.random.random() < 0.3:
            warp_factor = np.random.uniform(0.9, 1.1)
            new_len = int(n_time * warp_factor)
            if new_len > 2:
                # Interpolate each frequency band
                from scipy.ndimage import zoom
                feat = zoom(feat, (1.0, new_len / n_time), order=1)
                # Pad or truncate back to original length
                if feat.shape[1] < n_time:
                    pad_width = n_time - feat.shape[1]
                    feat = np.pad(feat, ((0, 0), (0, pad_width)), mode="constant")
                else:
                    feat = feat[:, :n_time]

        # Gaussian noise injection
        if np.random.random() < 0.3:
            feat = feat + np.random.normal(0, 0.03, feat.shape)

        # Random gain perturbation (±5%)
        if np.random.random() < 0.4:
            gain = np.random.uniform(0.95, 1.05)
            feat = feat * gain

        return feat

    def __getitem__(self, idx):
        audio_path, clarity = self.samples[idx]

        cache_path = self._get_mfcc_cache_path(audio_path)
        if os.path.exists(cache_path):
            mfcc = np.load(cache_path)
        else:
            result = preprocess_audio(audio_path)
            if result is None:
                return torch.zeros((1, 120, 200)), torch.tensor(clarity)
            signal, sr = result
            mfcc = extract_mfcc(signal, sr)
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
            np.save(cache_path, mfcc)

        if self.augment:
            mfcc = self._spec_augment(mfcc)

        X = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(clarity, dtype=torch.float32)

        return X, y