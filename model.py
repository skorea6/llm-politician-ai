import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


# LLM 스트리밍
def generate_stream(prompt: str):
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=256,
        temperature=0.0
    )

    for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta

        # content만 스트리밍
        text = getattr(delta, "content", None)
        if text:
            yield text


# 소규모 LLM
def run_small_llm(prompt: str, max_new_tokens: int = 16) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0.0
    )

    return response.choices[0].message.content.strip()

# 추후 로컬 LLM 사용시 아래 주석 해제
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.getenv("HUGGINGFACE_TOKEN"))
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     token=os.getenv("HUGGINGFACE_TOKEN")
# )
#
# print(model.device)
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.memory_summary())
#
# def generate_stream(prompt: str):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#
#     streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
#     thread = threading.Thread(target=lambda: model.generate(
#         **inputs,
#         max_new_tokens=128,
#         do_sample=False,
#         streamer=streamer,
#         # top_p=0.95,  # do_sample=False면 영향 거의 없음
#         # temperature=0.2,  # 매우 보수적
#         repetition_penalty=1.25,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id
#     ))
#     thread.start()
#
#     # streamer에서 토큰이 나오는 즉시 yield
#     for new_text in streamer:
#         yield new_text
#
#
# def run_small_llm(prompt: str, max_new_tokens: int = 16) -> str:
#     # 1) 입력 토큰화
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#
#     # 2) 프롬프트 길이 저장
#     prompt_len = inputs.input_ids.shape[1]
#
#     # 3) LLM 호출
#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             # temperature=0.0,
#             top_p=1.0,
#             eos_token_id=tokenizer.eos_token_id
#         )
#
#     # 4) prompt 이후 생성된 토큰만 디코딩
#     generated_ids = output[0][prompt_len:]
#     result = tokenizer.decode(generated_ids, skip_special_tokens=True)
#
#     return result.strip()
