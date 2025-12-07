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
        # top_p=0.95,  # do_sample=False면 영향 거의 없음
        # temperature=0.2,  # 매우 보수적
        repetition_penalty=1.25,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    ))
    thread.start()

    # streamer에서 토큰이 나오는 즉시 yield
    for new_text in streamer:
        yield new_text


def run_small_llm(prompt: str, max_new_tokens: int = 16) -> str:
    # 1) 입력 토큰화
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 2) 프롬프트 길이 저장
    prompt_len = inputs.input_ids.shape[1]

    # 3) LLM 호출
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # temperature=0.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id
        )

    # 4) prompt 이후 생성된 토큰만 디코딩
    generated_ids = output[0][prompt_len:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return result.strip()
