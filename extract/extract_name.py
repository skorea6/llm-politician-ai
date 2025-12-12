import re
import json
from typing import List

from model import run_small_llm


def _clean_name(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r'^[`\'"\(\[]+|[`\'"\)\]]+$', '', s)
    s = re.sub(r'(님|씨|군|양|선생님|의)$', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def _extract_json_array(text: str) -> List[str]:
    # 첫 번째 JSON 배열만 정확히 추출
    m = re.search(r"\[[^\]]*\]", text)
    if not m:
        return []

    try:
        arr = json.loads(m.group(0))
        return [_clean_name(x) for x in arr]
    except:
        return []


def extract_name_from_text(query: str, max_names: int = 3) -> List[str]:
    """
    LLM 전용 이름 추출
    - query: 사용자 입력 문장
    - max_names: 반환 최대 개수
    """
    prompt = f"""
규칙은 아래와 같습니다.
- 설명하지 말 것
- 이름만 JSON 배열로 출력
- JSON 배열 외 어떤 텍스트도 출력 금지
- 다른 질문, 답변을 만들어내지 마세요. 프롬프트에서 알려준 질문만 답변하세요.
- 다른 단어, 직업, 학력, 경력 등 출력 금지
- 최대 {max_names}명까지만

입력 문장에서 한국 사람 이름만 추출하고,
다음 형식으로만 출력하라: ["이름1", "이름2"]

입력 문장: "{query}"
출력:

JSON 배열만 출력하세요. 다른 글자는 절대 출력하지 마세요. 사람 이름이 없으면 []만 출력하세요.
"""
    raw = run_small_llm(prompt)
    print("raw 데이터: ")
    print(raw)
    names = _extract_json_array(raw)
    print("names: ")
    print(names)

    # 중복 제거 및 최대 개수 제한
    unique = []
    for n in names:
        if n not in unique:
            unique.append(n)
        if len(unique) >= max_names:
            break
    return unique

