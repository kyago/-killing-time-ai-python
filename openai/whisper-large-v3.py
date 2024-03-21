import torch
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

# speech to text

def find_first_mp3(directory):
    # 디렉토리 내 모든 파일 목록 가져오기
    files = os.listdir(directory)

    # .mp3 확장자를 가진 파일들만 걸러내기
    mp3_files = [file for file in files if file.endswith('.mp3')]

    if not mp3_files:
        return None, None  # 만약 .mp3 파일이 없으면 None 반환

    # 파일명을 기준으로 내림차순으로 정렬
    mp3_files.sort(reverse=True)

    # 첫 번째 .mp3 파일의 경로와 파일명 반환
    first_mp3_filename = mp3_files[0]
    first_mp3_path = os.path.join(directory, first_mp3_filename)

    return first_mp3_path, first_mp3_filename

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")

first_mp3_path, first_mp3_filename = find_first_mp3('../')

if first_mp3_path:
    print(".mp3 파일:", first_mp3_path)
    result = pipe(first_mp3_path)
    print(result["text"])
else:
    print("디렉토리에 .mp3 파일이 없습니다.")
