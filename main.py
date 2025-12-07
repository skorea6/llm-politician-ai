import os
import asyncio

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager

from build_rag_prompt import build_rag_prompt
from embedder import embed
from rank.rank import calc_additional_score
from update_politicians import update_politicians_daily
from store import init_collection, search_vectors, retrieve_by_id
from model import generate_stream
from config import QDRANT_COLLECTION_BASIC, QDRANT_COLLECTION_DETAIL
from dotenv import load_dotenv
load_dotenv()

AI_SECRET = os.getenv("AI_SECRET_KEY")

def verify_auth(x_ai_key: str = Header(None)):
    if x_ai_key != AI_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

###############################################
# 앱 시작 시 1회 + 매일 자동 업데이트
###############################################
scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    print("⚡ FastAPI STARTUP")

    init_collection()

    # 스케줄러 시작 (매일 3시 실행)
    # scheduler.add_job(update_politicians_daily, "cron", hour=3, minute=0)
    # scheduler.start()

    # asyncio.create_task(run_update())

    yield   # <--- 여기까지가 startup

    # --- shutdown ---
    print("FastAPI SHUTDOWN")
    scheduler.shutdown()

async def run_update():
    await asyncio.sleep(2)
    update_politicians_daily()

app = FastAPI(lifespan=lifespan)

###############################################
# LLM 모델 호출 (스프링이 호출)
###############################################
@app.post("/answer", dependencies=[Depends(verify_auth)])
async def answer(payload: dict):
    user_query = payload.get("query", "")
    if not user_query:
        return StreamingResponse(
            iter(["query가 없습니다.\n"]),
            media_type="text/plain"
        )


    # 1) 쿼리 임베딩 생성
    query_vec = embed(user_query)

    ###########################################################
    # 2) 1차 검색 — 기본 컬렉션(basic)에서 후보 N명 가져오기
    ###########################################################
    BASIC_LIMIT = 10
    basic_results = search_vectors(QDRANT_COLLECTION_BASIC, query_vec, limit=BASIC_LIMIT)

    candidates = []   # rerank 대상 ( {item, full_payload} 리스트 )

    if basic_results:
        # basic에서 나온 정치인 ID들에 대해 full json(keys: electors 포함) 조회
        for res in basic_results:
            pid = res.id

            detail_payload = retrieve_by_id(QDRANT_COLLECTION_DETAIL, pid)
            if not detail_payload:
                # detail 컬렉션에 없으면 일단 basic payload라도 사용
                full_payload = res.payload
            else:
                full_payload = (
                    detail_payload.get("full_payload")
                    if isinstance(detail_payload, dict) and "full_payload" in detail_payload
                    else detail_payload
                )

            candidates.append({
                "item": res,             # basic 검색 결과(embedding score 포함)
                "full": full_payload     # electors 포함 전체 JSON
            })

    ###########################################################
    # 3) fallback — basic 결과가 전혀 없으면 detail 컬렉션에서 직접 검색
    ###########################################################
    if not candidates:
        print("** 후보자를 찾지 못함 ** detail 검색 시작")
        DETAIL_LIMIT = 10
        detail_results = search_vectors(QDRANT_COLLECTION_DETAIL, query_vec, limit=DETAIL_LIMIT)

        if not detail_results:
            return StreamingResponse(
                iter(["관련 정치인을 찾지 못했습니다.\n"]),
                media_type="text/plain"
            )

        for res in detail_results:
            payload = res.payload
            if isinstance(payload, dict) and "full_payload" in payload:
                full_payload = payload["full_payload"]
            else:
                full_payload = payload

            candidates.append({
                "item": res,
                "full": full_payload
            })

    ###########################################################
    # 4) elector + votePercentage 기반 rerank 적용
    ###########################################################
    reranked = []
    for c in candidates:
        base_score = c["item"].score
        full_payload = c["full"]

        bonus = calc_additional_score(full_payload)
        final_score = base_score + bonus

        reranked.append({
            "item": c["item"],
            "full": full_payload,
            "final_score": final_score
        })

    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    print(reranked)
    # 최종 top 1
    best = reranked[0]
    final_score = best["final_score"]
    full_payload = best["full"]

    ###########################################################
    # 5) threshold 확인 (너무 낮으면 실패 처리)
    ###########################################################
    if final_score < 0.2:
        return StreamingResponse(
            iter(["유사도 낮음: 관련 정치인을 찾지 못했습니다.\n"]),
            media_type="text/plain"
        )

    ############################################
    # 4) RAG 프롬프트 생성
    ############################################
    prompt = build_rag_prompt(user_query, [full_payload])

    # 4. LLM 스트리밍 응답
    def stream():
        for chunk in generate_stream(prompt):
            yield chunk + "\n"

    return StreamingResponse(stream(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
