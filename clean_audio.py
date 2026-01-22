import os
from mlservice.utils.audio_processing import preprocess_audio

# ---------------- CONFIG ----------------
DATASET_ROOT = "data/audio"

def clean_dataset(root_dir):
    deleted = 0
    checked = 0

    print("🔍 Starting silent-audio cleanup...\n")

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith(".wav"):
                continue

            path = os.path.join(root, file)
            checked += 1

            # preprocess_audio returns None if audio is silent/invalid
            result = preprocess_audio(
                path,
                delete_if_silent=True  # 🔴 PERMANENT DELETE
            )

            if result is None:
                deleted += 1

    print("\n✅ CLEANING COMPLETE")
    print(f"📁 Total files checked : {checked}")
    print(f"🗑 Files deleted       : {deleted}")
    print(f"✔ Remaining files     : {checked - deleted}")

if __name__ == "__main__":
    clean_dataset(DATASET_ROOT)
