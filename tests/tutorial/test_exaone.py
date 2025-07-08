from mblt_model_zoo.transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mobilint/EXAONE-3.5-2.4B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Choose your prompt
prompt = "Explain how wonderful you are"  # English example
prompt = "스스로를 자랑해 봐"       # Korean example

messages = [
    {"role": "system", 
     "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
    {"role": "user", "content": prompt}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

output = model.generate(
    input_ids,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=1024,
    do_sample=False,
)
print(tokenizer.decode(output[0]))
