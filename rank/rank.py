def calc_additional_score(payload: dict) -> float:
    """
    payload: 정치인 JSON 전체
    electors 배열 기반 가중치 계산
    - electors 개수 × 0.5점
    - winner 있으면 +1점
    - votePercentage(0~100)를 0~1로 정규화해 모두 합산
    """
    score = 0.0

    electors = payload.get("electors", [])
    if not isinstance(electors, list):
        return score  # 잘못된 구조면 0점

    # 1) electors 개수 × 0.5점
    score += len(electors) * 0.05

    # 2) winner 존재하면 +1점
    for e in electors:
        if isinstance(e, dict) and e.get("winner") is True:
            score += 0.1
            break

    # 3) votePercentage 점수: 0~100 → 0~1 변환해서 합산
    for e in electors:
        if not isinstance(e, dict):
            continue

        vp = e.get("votePercentage")
        if isinstance(vp, (int, float)):
            # 0~1로 정규화하여 점수에 추가
            score += max(0.0, min(vp / 100.0, 1.0) / 5)

    return score