import os
import shutil
import random

RAW_ROOT = "als"
TARGET_ROOT = "data/audio"
TRAIN_SPLIT = 0.8
ARRAYMIC_RATIO = 0.3
SEED = 42

random.seed(SEED)

LABEL_MAP = {
    "F": "dysarthric",
    "M": "dysarthric",
    "FC": "normal",
    "MC": "normal"
}

# Create target directories
for split in ["train", "test"]:
    for label in ["normal", "dysarthric"]:
        os.makedirs(f"{TARGET_ROOT}/{split}/{label}/headmic", exist_ok=True)
        if split == "train":
            os.makedirs(f"{TARGET_ROOT}/{split}/{label}/arraymic", exist_ok=True)

print("✔ Target directories created")

# Process dataset
for group, label in LABEL_MAP.items():
    group_path = os.path.join(RAW_ROOT, group)
    if not os.path.isdir(group_path):
        continue

    headmic_files = []
    arraymic_files = []

    for speaker in os.listdir(group_path):
        speaker_path = os.path.join(group_path, speaker)
        if not os.path.isdir(speaker_path):
            continue  # skips .DS_Store safely

        for session in os.listdir(speaker_path):
            session_path = os.path.join(speaker_path, session)
            if not os.path.isdir(session_path):
                continue

            headmic_path = os.path.join(session_path, "wav_headMic")
            arraymic_path = os.path.join(session_path, "wav_arrayMic")

            if os.path.isdir(headmic_path):
                for f in os.listdir(headmic_path):
                    if f.endswith(".wav"):
                        headmic_files.append(
                            (os.path.join(headmic_path, f), f"{speaker}_{f}")
                        )

            if os.path.isdir(arraymic_path):
                for f in os.listdir(arraymic_path):
                    if f.endswith(".wav"):
                        arraymic_files.append(
                            (os.path.join(arraymic_path, f), f"{speaker}_{f}")
                        )

    random.shuffle(headmic_files)
    split_idx = int(len(headmic_files) * TRAIN_SPLIT)

    train_headmic = headmic_files[:split_idx]
    test_headmic = headmic_files[split_idx:]

    # Copy headmic (train + test)
    for src, name in train_headmic:
        shutil.copy(src, f"{TARGET_ROOT}/train/{label}/headmic/{name}")

    for src, name in test_headmic:
        shutil.copy(src, f"{TARGET_ROOT}/test/{label}/headmic/{name}")

    # Copy limited arraymic (train only)
    random.shuffle(arraymic_files)
    max_array = int(len(train_headmic) * ARRAYMIC_RATIO)

    for src, name in arraymic_files[:max_array]:
        shutil.copy(src, f"{TARGET_ROOT}/train/{label}/arraymic/{name}")

    print(
        f"✔ {label.upper()} | "
        f"HeadMic train: {len(train_headmic)}, "
        f"HeadMic test: {len(test_headmic)}, "
        f"ArrayMic train: {max_array}"
    )

print("\n🎉 Dataset preparation completed successfully!")
