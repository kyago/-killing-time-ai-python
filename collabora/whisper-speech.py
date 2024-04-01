import torch
from whisperspeech.pipeline import Pipeline

# collabora/WhisperSpeech Does it support other languages? -> 한국어 미지원. https://huggingface.co/spaces/collabora/WhisperSpeech/discussions/1

# GPU 디바이스 Apple silicon mps 설정
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# let's start with the fast SD S2A model
pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')

audio_object = pipe.generate_to_file("../whisperspeech_output.wav", "This is the first demo of Whisper Speech, a fully open source text-to-speech model trained by Collabora and Lion on the Juwels supercomputer.")