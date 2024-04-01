import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = '01-ai/Yi-9B'
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch_dtype
).eval()

# Prompt content: "hi"
messages = [
    {"role": "user", "content": "hi"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to(device))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)
