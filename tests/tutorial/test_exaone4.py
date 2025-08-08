from mblt_model_zoo.transformers import pipeline, AutoTokenizer
from transformers import TextStreamer
import random

model_name = "mobilint/EXAONE-4.0-1.2B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline(
    "text-generation",
    model=model_name,
    streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
)

# Non-reasoning mode

# Choose your prompt
prompt = "Explain how wonderful you are"
prompt = "Explica lo increíble que eres"
prompt = "너가 얼마나 대단한지 설명해 봐"

messages = [
    {"role": "user", "content": prompt}
]

pipe(
    messages,
    max_new_tokens=128,
    do_sample=False,
)

# Reasoning mode

messages = [
    {"role": "user", "content": "Which one is bigger, 3.12 vs 3.9?"}
]

pipe(
    messages,
    enable_thinking=True,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.6,
    top_p=0.95
)

# Agentic tool use

def roll_dice(max_num: int):
    return random.randint(1, max_num)

tools = [
    {
        "type": "function",
        "function": {
            "name": "roll_dice",
            "description": "Roll a dice with the number 1 to N. User can select the number N.",
            "parameters": {
                "type": "object",
                "required": ["max_num"],
                "properties": {
                    "max_num": {
                        "type": "int",
                        "description": "Max number of the dice"
                    }
                }
            }
        }
    }
]

messages = [
    {"role": "user", "content": "Roll D6 dice twice!"}
]

pipe(
    messages,
    tools=tools,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.6,
    top_p=0.95
)
