import requests
from datetime import datetime
import json

from config import POLITICIAN_API_URL, QDRANT_COLLECTION_BASIC, QDRANT_COLLECTION_DETAIL
from embedder import embed
from store import upsert_batch


###############################################
# 정치인 API 수집 (매일 1회)
###############################################
def fetch_page(page):
    res = requests.post(POLITICIAN_API_URL, json={"page": page})
    res.raise_for_status()
    return res.json().get("data", [])


JSON_SCHEMA_DESCRIPTION = """
이 데이터는 대한민국 정치인의 상세정보 JSON 입니다.
각 필드의 의미는 아래와 같습니다:

- politicianId: 정치인 고유 ID
- name: 이름
- gender: 성별(MAN: 남자, WOMAN: 여자)
- birthDate: 생년월일
- address: 거주지 (실제로 살고 있는 지역)
- job: 직업 또는 직함 (줄바꿈으로 구분되어있음)
- career: 주요 경력 (줄바꿈으로 구분되어있음. 앞에 (현)이 붙여지면 현재 하고 있는것.)
- education: 학력 (줄바꿈으로 구분되어있음)
- criminalRecord: 전과 여부(숫자만. 0=없음)
- image: 데이터 무시(ignore)
- politicalParty: 정치인의 정당 정보
    - politicalPartyId: 정당 고유 ID
    - name: 정당 이름
    - countMembers: 정당에 들어있는 국회의원 수
    - foundYear: 정당 설립일
    - representativeName: 정당 대표 이름
    - personalColor: 정당 대표 색깔 (RGB 코드)
    - logoImage: 데이터 무시(ignore)
    - coverImage: 데이터 무시(ignore)
- electors: 정치인의 선거 출마 기록 배열 (여러개 올수있음)
    - electorId: 선거 고유 ID
    - electionType:
        - electionTypeId: 선거 타입 고유 ID
        - electionMainType: CONGRESS_MAN(국회의원) 혹은 PRESIDENT(대통령) 올 수 있음
        - electionSubType: CONGRESS_PRESIDENT_MAN(국회의원+대통령) 혹은 CONGRESS_MAN(국회의원) 혹은 PRESIDENT(대통령) 올 수 있음
        - electionDate: 선거(투표) 일자
        - round: 몇대 선거 (예: 숫자 22라면, 22대 선거)
    - electorTypes: 배열 형태로 PRELIMINARY_CANDIDATE(예비 후보자), CANDIDATE(후보자), ELECTION_WINNER(당선자)가 올 수 있음. 여러개 올수있음
    - politicalParty: 정당 정보
        - politicalPartyId: 정당 고유 ID
        - name: 정당 이름
        - countMembers: 정당에 들어있는 국회의원 수
        - foundYear: 정당 설립일
        - representativeName: 정당 대표 이름
        - personalColor: 정당 대표 색깔 (RGB 코드)
        - logoImage: 데이터 무시(ignore)
        - coverImage: 데이터 무시(ignore)
    - zoneElectionDistrict: 선거 지역
        - zoneElectionDistrictId: 선거 지역 고유ID
        - zoneCity: 도시
            - zoneCityId: 도시 고유ID
            - name: 도시 이름
        - name: 지역(구시군) 이름
        - peopleCount: 해당 선거 지역의 사람 총원 수 (투표할 수 있는 총원)
        - totalVoteCount: 총 투표 수 [선거 지역기준]
        - realVoteCount: 실제 정상적으로 투표된 수 [선거 지역기준]
        - ignoredVoteCount: 무효 투표 수 [선거 지역기준]
        - abandonedVoteCount: 기권 투표 수 [선거 지역기준]
    - informationUrl: 참고 URL (해당 선거 정보를 자세히 보고 싶다면 여기로 안내)
    - preliminaryRegisteredDate: 예비후보자 등록 날짜
    - electionNum: 후보자가 기호 몇번인지
    - winner: true or false (당선 되었는지 안되었는지)
    - voteCount: 후보자가 받은 투표 수
    - votePercentage: 몇퍼센트로 투표 받았는지
- ranking: 데이터 무시(ignore)
- follower: 데이터 무시(ignore)

이 JSON 전체를 그대로 이해해서 검색 가능한 형태로 임베딩해야 한다.
"""

def make_basic_text(name, core):
    prefix = f"이 데이터는 대한민국 정치인 '{name}'의 핵심 요약 정보입니다.\n"
    return prefix + json.dumps(core, ensure_ascii=False)

def make_detail_payload(p):
    """
    상세 컬렉션에 저장할 payload. 전체 원본을 넣음.
    """
    return {"full_payload": p}

def update_politicians_daily():
    print("\n[UPDATE START]", datetime.now())

    page = 1
    total = 0
    batch_basic = []
    batch_detail = []
    BATCH_SIZE = 200

    while True:
        data = fetch_page(page)
        if not data:
            break

        for p in data:
            pid = int(p["politicianId"])

            core = {
                "id": pid,
                "name": p.get("name"),
                "birthDate": p.get("birthDate"),
                "address": p.get("address"),
                "gender": p.get("gender"),
                "criminalRecord": p.get("criminalRecord"),
                "education": "\\n".join(
                    p.get("education", []) if isinstance(p.get("education", []), list) else [p.get("education", "")]),
                "job": "\\n".join(
                    p.get("job", []) if isinstance(p.get("job", []), list) else [p.get("job", "")]),
                "career": "\\n".join(
                    p.get("career", []) if isinstance(p.get("career", []), list) else [p.get("career", "")]),
                "short_bio": f"정치인 이름이 '{p.get('name', '')}'인 사람의 성별은 {p.get('gender', '')}이고 생년월일은 {p.get('birthDate', '')}이다. 사는 곳은 {p.get('address', '')}이다. 범죄 기록은 {p.get('criminalRecord', '')}건이다."
            }

            # 1) 기본(검색)용 임베딩
            embedding_text = make_basic_text(p.get("name"), core)
            vector_text = embed(embedding_text)

            embedding_detail_text = make_basic_text(p.get("name"), p)
            vector_detail = embed(embedding_detail_text)

            vector_name = embed(p.get("name", ""))

            basic_point = {
                "id": pid,
                "vector": {
                    "text_vector": vector_text,
                    "name_vector": vector_name
                },
                "payload": core
            }

            # 2) 상세(원본) 저장
            detail_point = {
                "id": pid,
                "vector": vector_detail,
                "payload": make_detail_payload(p)
            }

            batch_basic.append(basic_point)
            batch_detail.append(detail_point)

            total += 1

            if len(batch_basic) >= BATCH_SIZE:
                upsert_batch(QDRANT_COLLECTION_BASIC, batch_basic)
                print(f"[BASIC BATCH] {len(batch_basic)} 업로드 완료")
                batch_basic = []

            if len(batch_detail) >= BATCH_SIZE:
                upsert_batch(QDRANT_COLLECTION_DETAIL, batch_detail)
                print(f"[DETAIL BATCH] {len(batch_detail)} 업로드 완료")
                batch_detail = []

        page += 1

    # 마지막 남은 배치 처리
    if batch_basic:
        upsert_batch(QDRANT_COLLECTION_BASIC, batch_basic)
        print(f"[BASIC BATCH] 마지막 {len(batch_basic)} 업로드 완료")
    if batch_detail:
        upsert_batch(QDRANT_COLLECTION_DETAIL, batch_detail)
        print(f"[DETAIL BATCH] 마지막 {len(batch_detail)} 업로드 완료")

    print(f"[UPDATE] 총 {total}명 정치인 업로드 완료")