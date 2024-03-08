from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")

input_text = "write me a poem about beautifuly woman."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))