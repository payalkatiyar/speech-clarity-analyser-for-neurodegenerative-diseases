import os
import shutil
from mlservice.utils.audio_processing import preprocess_audio

# ---------------- CONFIG ----------------
DATASET_ROOT = "data/audio"
QUARANTINE_ROOT = "data/quarantine"

def clean_dataset(root_dir):
    moved = 0
    checked = 0

    print("🔍 Starting silent-audio cleanup...\n")

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith(".wav"):
                continue

            path = os.path.join(root, file)
            checked += 1

            # Check audio WITHOUT deleting
            result = preprocess_audio(
                path,
                delete_if_silent=False
            )

            if result is None:
                # Preserve relative structure in quarantine
                rel_path = os.path.relpath(path, DATASET_ROOT)
                quarantine_path = os.path.join(QUARANTINE_ROOT, rel_path)

                os.makedirs(os.path.dirname(quarantine_path), exist_ok=True)
                shutil.move(path, quarantine_path)

                moved += 1

    print("\n✅ CLEANING COMPLETE")
    print(f"📁 Total files checked : {checked}")
    print(f"🚫 Files quarantined   : {moved}")
    print(f"✔ Remaining files     : {checked - moved}")

if __name__ == "__main__":
    clean_dataset(DATASET_ROOT)
