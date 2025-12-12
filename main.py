import os
import asyncio

from qdrant_client.models import Filter, FieldCondition, MatchAny, NamedVector
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager

from build_rag_prompt import build_rag_prompt
from embedder import embed
from extract.extract_name import extract_name_from_text
from rank.rank import calc_additional_score
from update_politicians import update_politicians_daily
from store import init_collection, search_vectors, retrieve_by_id, qdrant
from model import generate_stream
from config import QDRANT_COLLECTION_BASIC, QDRANT_COLLECTION_DETAIL
from dotenv import load_dotenv
load_dotenv()

AI_SECRET = os.getenv("AI_SECRET_KEY")

def verify_auth(x_ai_key: str = Header(None)):
    if x_ai_key != AI_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

# 앱 시작 시 1회 + 매일 자동 업데이트
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

# LLM 모델 호출 (스프링 webflux가 호출)
@app.post("/answer", dependencies=[Depends(verify_auth)])
async def answer(payload: dict):
    user_query = payload.get("query", "")
    if not user_query:
        return StreamingResponse(
            iter(["query가 없습니다.\n"]),
            media_type="text/plain"
        )


    # 쿼리 임베딩
    query_vec = embed(user_query)

    # 1) 이름 전용 임베딩 생성
    names = extract_name_from_text(user_query, max_names=3)
    print(names)

    name_ids = []

    if names:
        for nm in names:
            vec = embed(nm)

            # 2) name_vector로 먼저 검색 (이름 우선 검색)
            name_results = qdrant.search(
                collection_name=QDRANT_COLLECTION_BASIC,
                query_vector=NamedVector(name="name_vector", vector=vec),
                limit=5
            )

            found_ids = [r.id for r in name_results]
            name_ids.extend(found_ids)

    name_ids = list(set(name_ids))
    candidates = []

    # 3) 이름 후보가 있으면 content_vector + filter 검색
    if name_ids:
        print("이름 후보 찾음!!")
        print(name_ids)
        filtered = qdrant.search(
            collection_name=QDRANT_COLLECTION_BASIC,
            query_vector=NamedVector(name="text_vector", vector=query_vec),
            limit=15,
            query_filter=Filter(
                must=[FieldCondition(
                    key="id",
                    match=MatchAny(any=name_ids)
                )]
            )
        )

        for res in filtered:
            pid = res.id
            detail_payload = retrieve_by_id(QDRANT_COLLECTION_DETAIL, pid)
            full_payload = detail_payload.get("full_payload", detail_payload)
            candidates.append({"item": res, "full": full_payload})

            print(pid)

    # 4) 이름 기반 후보 부족 → BASIC 일반 검색
    if len(candidates) < 3:
        basic_results = search_vectors(QDRANT_COLLECTION_BASIC, query_vec, limit=5)
        for res in basic_results:
            pid = res.id
            detail_payload = retrieve_by_id(QDRANT_COLLECTION_DETAIL, pid)
            full_payload = detail_payload.get("full_payload", detail_payload)
            candidates.append({"item": res, "full": full_payload})

    # 5) basic 결과가 전혀 없으면 detail 컬렉션에서 직접 검색
    if not candidates:
        print("** 후보자를 찾지 못함 ** detail 검색 시작")
        DETAIL_LIMIT = 5
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

    # 6) elector + votePercentage 기반 rerank 적용
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

    filtered = [c for c in reranked if c["final_score"] >= 0.2]
    if not filtered:
        return StreamingResponse(
            iter(["유사도 낮음: 관련 정치인을 찾지 못했습니다.\n"]),
            media_type="text/plain"
        )

    top_k = filtered[:3]
    full_payload = [c["full"] for c in top_k]

    # 7) RAG 프롬프트 생성
    prompt = build_rag_prompt(user_query, [full_payload])

    # 8) LLM 스트리밍 응답
    def stream():
        for chunk in generate_stream(prompt):
            yield chunk + "\n"

    return StreamingResponse(stream(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
