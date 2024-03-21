from transformers import VitsModel, AutoTokenizer
import torch
import uroman
from datetime import datetime

# model = VitsModel.from_pretrained("facebook/mms-tts-eng")
# tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

model = VitsModel.from_pretrained("facebook/mms-tts-kor")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")

text = "한글로 말하는 기능에 대한 테스트. 문장을 매끄럽게 얘기할 수 있는지? 이런 텍스트를 자연스럽게 읽어 줄 수 있으려나?"
text_romanized = uroman.uroman(text, uroman.Language.get('ko-KR').language)  # 한글 문자열을 로마자로 변환
inputs = tokenizer(text_romanized, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

import scipy
now = datetime.now()
dt_string = now.strftime("%d%m%Y%H%M%S")
filename = "../" + dt_string + "_tts_output.wav"

scipy.io.wavfile.write(filename, rate=model.config.sampling_rate, data=output.T.float().numpy())