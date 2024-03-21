from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("beomi/gemma-ko-7b")
model = AutoModelForCausalLM.from_pretrained("beomi/gemma-ko-7b")

input_text = "머신러닝과 딥러닝의 차이는?"
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
