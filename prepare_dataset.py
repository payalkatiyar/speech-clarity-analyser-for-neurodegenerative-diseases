import os
import shutil
import random

# ================= CONFIG =================
RAW_ROOT = "als"
TARGET_ROOT = "data/audio"

TRAIN_SPLIT = 0.8          # speaker-level split
ARRAYMIC_RATIO = 0.3       # % of headmic count
SEED = 42

random.seed(SEED)

LABEL_MAP = {
    "F": "dysarthric",
    "M": "dysarthric",
    "FC": "normal",
    "MC": "normal"
}

MIC_FOLDERS = {
    "headmic": "wav_headMic",
    "arraymic": "wav_arrayMic"
}

# ================= CREATE TARGET DIRS =================
for split in ["train", "test"]:
    for label in ["normal", "dysarthric"]:
        os.makedirs(f"{TARGET_ROOT}/{split}/{label}/headmic", exist_ok=True)
        if split == "train":
            os.makedirs(f"{TARGET_ROOT}/{split}/{label}/arraymic", exist_ok=True)

print("✔ Target directories created")

# ================= PROCESS EACH GROUP =================
for group, label in LABEL_MAP.items():
    group_path = os.path.join(RAW_ROOT, group)
    if not os.path.isdir(group_path):
        continue

    speakers = [
        s for s in os.listdir(group_path)
        if os.path.isdir(os.path.join(group_path, s))
    ]

    random.shuffle(speakers)
    split_idx = int(len(speakers) * TRAIN_SPLIT)

    train_speakers = speakers[:split_idx]
    test_speakers = speakers[split_idx:]

    def process_speakers(speaker_list, split):
        headmic_count = 0
        arraymic_files = []

        for speaker in speaker_list:
            speaker_path = os.path.join(group_path, speaker)

            for session in os.listdir(speaker_path):
                if not session.lower().startswith("session"):
                    continue

                session_path = os.path.join(speaker_path, session)

                # -------- HeadMic --------
                headmic_path = os.path.join(
                    session_path, MIC_FOLDERS["headmic"]
                )
                if os.path.isdir(headmic_path):
                    for wav in os.listdir(headmic_path):
                        if not wav.endswith(".wav"):
                            continue

                        src = os.path.join(headmic_path, wav)
                        dst = os.path.join(
                            TARGET_ROOT,
                            split,
                            label,
                            "headmic",
                            f"{speaker}_{session}_{wav}"
                        )
                        shutil.copy(src, dst)
                        headmic_count += 1

                # -------- ArrayMic (train only) --------
                if split == "train":
                    arraymic_path = os.path.join(
                        session_path, MIC_FOLDERS["arraymic"]
                    )
                    if os.path.isdir(arraymic_path):
                        for wav in os.listdir(arraymic_path):
                            if wav.endswith(".wav"):
                                arraymic_files.append(
                                    (
                                        os.path.join(arraymic_path, wav),
                                        f"{speaker}_{session}_{wav}"
                                    )
                                )

        # Limit arraymic usage (robustness without domination)
        if split == "train":
            random.shuffle(arraymic_files)
            max_array = int(headmic_count * ARRAYMIC_RATIO)

            for src, name in arraymic_files[:max_array]:
                dst = os.path.join(
                    TARGET_ROOT,
                    "train",
                    label,
                    "arraymic",
                    name
                )
                shutil.copy(src, dst)

            return headmic_count, max_array

        return headmic_count, 0

    train_head, train_array = process_speakers(train_speakers, "train")
    test_head, _ = process_speakers(test_speakers, "test")

    print(
        f"✔ {label.upper()} | "
        f"Train speakers: {len(train_speakers)}, "
        f"Test speakers: {len(test_speakers)}, "
        f"HeadMic train: {train_head}, "
        f"HeadMic test: {test_head}, "
        f"ArrayMic train: {train_array}"
    )

print("\n🎉 Dataset preparation completed successfully (speaker-safe)")
