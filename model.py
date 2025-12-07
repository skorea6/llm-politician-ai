import os
import threading

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch

from config import MODEL_NAME
from dotenv import load_dotenv
load_dotenv()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.getenv("HUGGINGFACE_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    token=os.getenv("HUGGINGFACE_TOKEN")
)

print(model.device)
print(torch.cuda.get_device_name(0))
print(torch.cuda.memory_summary())

def generate_stream(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    thread = threading.Thread(target=lambda: model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        streamer=streamer,
        top_p=0.95,  # do_sample=False면 영향 거의 없음
        temperature=0.2,  # 매우 보수적
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    ))
    thread.start()

    # streamer에서 토큰이 나오는 즉시 yield
    for new_text in streamer:
        yield new_text
