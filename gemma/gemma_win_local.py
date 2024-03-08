from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "E:/models/2b-gemma"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

input_text = "write me a poem about beautifuly woman."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))