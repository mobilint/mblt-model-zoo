from mblt_model_zoo.transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mobilint/HyperCLOVAX-SEED-Text-Instruct-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

chat = [
    {"role": "tool_list", "content": ""},
    {
        "role": "system",
        "content": '- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n- 오늘은 2025년 04월 24일(목)이다.',
    },
    {
        "role": "user",
        "content": "슈뢰딩거 방정식과 양자역학의 관계를 최대한 자세히 알려줘.",
    },
]

inputs = tokenizer.apply_chat_template(
    chat, add_generation_prompt=True, return_dict=True, return_tensors="pt"
)
output_ids = model.generate(
    **inputs,
    max_length=1024,
    stop_strings=["<|endofturn|>", "<|stop|>"],
    tokenizer=tokenizer,
)
print(tokenizer.batch_decode(output_ids)[0])
