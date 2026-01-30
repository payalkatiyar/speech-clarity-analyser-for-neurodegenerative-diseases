import os
import torch
from torch.utils.data import Dataset
from collections import defaultdict

from mlservice.utils.audio_processing import preprocess_audio, extract_mfcc
from mlservice.clarity_labels import raw_clarity_score, normalize_scores_within_speaker


class SpeechDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        speaker_buckets = defaultdict(list)

        # ---------- COLLECT FILES ----------
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

                    # speaker id from filename (F01_XXXX.wav)
                    speaker = file.split("_")[0]

                    session = "session1"
                    name = file.lower()
                    if "session2" in name:
                        session = "session2"
                    elif "session3" in name:
                        session = "session3"

                    path = os.path.join(mic_dir, file)

                    speaker_buckets[speaker].append(
                        (path, label, session)
                    )

        # ---------- COMPUTE SPEAKER-RELATIVE SCORES ----------
        for speaker, items in speaker_buckets.items():
            raw_scores = [
                raw_clarity_score(label, session, path)
                for path, label, session in items
            ]

            norm_scores = normalize_scores_within_speaker(raw_scores)

            for (path, label, session), score in zip(items, norm_scores):
                self.samples.append((path, score))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, clarity = self.samples[idx]

        result = preprocess_audio(audio_path)
        if result is None:
            return self.__getitem__((idx + 1) % len(self.samples))

        signal, sr = result
        mfcc = extract_mfcc(signal, sr)

        X = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(clarity, dtype=torch.float32)

        return X, y
