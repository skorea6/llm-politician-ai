import json

def build_rag_prompt(question: str, hits) -> str:
    context = ""
    for h in hits:
        # h은 full politician json
        context += "정치인 정보:\n" + json.dumps(h, ensure_ascii=False) + "\n\n"

    prompt = f"""
1. 당신은 정치인 정보를 답변하는 AI 모델입니다.

2. 아래 정치인 정보(JSON)들만 참고해 답하십시오:
{context}

3. 각 필드의 의미는 아래와 같습니다:
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

4. 규칙:
- json에 등장하는 정보만 사용
- json에 정보가 없으면 "잘 모르겠습니다."
- 절대 새로운 질문을 만들지 마세요
- 답변은 한 문장으로만 출력
- HTML 태그, <p>, <hr> 등 포함 금지

5. 질문: {question}
답변:
""".strip()

    print(prompt)

    return prompt
