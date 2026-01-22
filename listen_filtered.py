from mlservice.utils.audio_processing import save_filtered_audio

# 🔽 CHANGE THIS PATH TO ANY WAV YOU WANT TO HEAR
INPUT_AUDIO = "data/audio/train/normal/headmic/FC01_0130.wav"
save_filtered_audio(INPUT_AUDIO)
INPUT_AUDIO = "data/audio/test/normal/headmic/FC01_0077.wav"
save_filtered_audio(INPUT_AUDIO)
INPUT_AUDIO = "data/audio/test/normal/headmic/MC04_0493.wav"
save_filtered_audio(INPUT_AUDIO)
INPUT_AUDIO = "data/audio/test/normal/headmic/MC04_0466.wav"
save_filtered_audio(INPUT_AUDIO)
INPUT_AUDIO = "data/audio/test/normal/headmic/MC04_0646.wav"
save_filtered_audio(INPUT_AUDIO)
INPUT_AUDIO = "data/audio/train/dysarthric/arraymic/F03_0303.wav"
save_filtered_audio(INPUT_AUDIO)
INPUT_AUDIO = "data/audio/train/dysarthric/arraymic/M04_0172.wav"
save_filtered_audio(INPUT_AUDIO)
