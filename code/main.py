"""
    # 1) 인덱스 빌드
    !python code/main.py build --pdf "data/전자금융거래법(법률)(제19734호)(20240915).pdf"
    !python code/main.py build --dir "laws/"
    !python code/main.py build --dir "supplement/"
    
    # 2) 단일 질문
    !python code/main.py ask --question "전자자금이체의 지급 효력 발생 시점을 전자금융거래법 기준에 따라 설명하세요."
    !python code/main.py ask --question "금융회사가 정보보호 예산을 관리할 때, 전자금융감독규정상 정보기술부문 인력 및 예산의 기준 비율은 얼마인가요?"
    
    !python code/main.py ask --question $'전자금융거래법 제44조에 따르면, 청문 절차가 필요한 경우는 무엇인가?\n1 전자금융거래의 중단\n2 전자금융거래의 보안 점검\n3 전자금융업자의 등록 취소\n4 전자금융거래의 수수료 변경'
    !python code/main.py ask --question $'국내대리인이 법을 위반한 경우, 그 책임은 누구에게 있는가?\n1 국내대리인\n2 정부기관\n3 법원\n4 정보통신서비스 제공자\n5 개인정보 처리 위탁업체'
    !python code/main.py ask --question $'개인정보보호법 제63조에 따르면, 보호위원회가 자료제출 요구 및 검사를 통해 수집한 서류나 자료를 제3자에게 제공하거나 일반에 공개할 수 있는 경우는?\n1 자료가 비밀이 아닌 경우\n2 개인정보처리자의 동의가 있는 경우\n3 정보주체가 개인정보 열람을 요청한 경우\n4 법에 따른 경우\n5 보호위원회의 내부 규정에 따른 경우'

    # 3) test.csv 한 번에 추론(컬럼명: Question)
    !python code/main.py run --csv "data/test.csv"

"""




import os
import torch
import pandas as pd
import argparse

from model import load_model
from text_utils import _ensure_dir
from rag import RAGIndexer, RAGRetriever, STReranker, answer_with_rag
from utils import extract_answer_only
from config import TOP_K, SCORE_THRESHOLD, M_GENERIC, M_FILTERED, OUTPUT_PATH, RERANK_THRESHOLD

INDEX_DIR = "./rag_index"
MODEL_NAME = "/workspace/models/multilingual-e5-small"

def cmd_build(args):
    _ensure_dir(INDEX_DIR)
    indexer = RAGIndexer(MODEL_NAME, device="cpu")

    pdfs = []
    if args.pdf:
        pdfs.extend(args.pdf)
    if args.dir:
        for name in os.listdir(args.dir):
            if name.lower().endswith(".pdf"):
                pdfs.append(os.path.join(args.dir, name))
    if not pdfs:
        raise ValueError("PDF가 없습니다. --pdf 다중 또는 --dir를 지정하세요.")

    # ★ 추가: --dir의 마지막 폴더명을 네임스페이스로 사용해 INDEX_DIR/<ns>에 저장
    if args.dir:
        ns = os.path.basename(os.path.normpath(args.dir))  # "laws/"
    else:
        ns = "default"
    out_dir = os.path.join(INDEX_DIR, ns)

    indexer.build_many(pdfs, index_dir=out_dir)   # ← 여기!



def cmd_ask(args):
    
    pipe = load_model()

    # 1) 두 인덱스 로드
    retrA = RAGRetriever(index_dir=os.path.join(INDEX_DIR, "laws"), device="cpu")
    retrB = RAGRetriever(index_dir=os.path.join(INDEX_DIR, "supplement"), device="cpu")

    # 2) Reranker 준비
    rr_model = getattr(args, "rerank_model", "/workspace/models/gte-multilingual-reranker-base")
    rr_device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = STReranker(model_name_or_path=rr_model, device=rr_device, max_length=800)

    # 3) 하이퍼파라미터
    top_k = getattr(args, "top_k", TOP_K)
    score_threshold = getattr(args, "threshold", SCORE_THRESHOLD)          # (폴백 경로용)
    rerank_threshold = getattr(args, "rerank_threshold", RERANK_THRESHOLD) # CE 컨텍스트 게이트
    M_generic_A = getattr(args, "M_generic", M_GENERIC)                    # laws 전역 후보
    M_filtered_A = getattr(args, "M_filtered", M_FILTERED)                 # laws 필터 후보
    M_generic_B = getattr(args, "M_generic_B", M_generic_A)                # supplement 전역 후보
    keep_for_ce = getattr(args, "keep_for_ce", 50)
    use_clause = getattr(args, "use_clause", True)

    q = args.question

    # 4) 혼합 검색(A 전역+필터, B 전역) → RRF → CE rerank
    #    멀티 인덱스 answer_with_rag은 meta_dbg를 추가로 반환
    prompt, gen, passages, hits, use_context, contexts, meta_dbg = answer_with_rag(
        q,
        retrieverA=retrA,
        retrieverB=retrB,
        pipe=pipe,
        top_k=top_k,
        score_threshold=score_threshold,           # (단일 폴백 경로에서만 사용)
        reranker=reranker,
        rerank_threshold=rerank_threshold,
        M_generic=M_generic_A,
        M_filtered=M_filtered_A,
        M_generic_B=M_generic_B,
        keep_for_ce=keep_for_ce,
        use_clause=use_clause,
    )

    # 5) 출력
    print("\n===== 생성된 답변 =====")
    print(gen)
    ans = extract_answer_only(gen, original_question=q, prompt=prompt)
    print("\n===== 채택된 답변 =====")
    print(ans)

    print("\n===== 검색 결과 요약 =====")
    if hits:
        # 멀티 인덱스 경로: hits는 ("A:idx", ce_score) 형태
        print(f"Top-1 ce_score={hits[0][1]:.3f} | rerank_threshold={rerank_threshold:.2f} | context_used={use_context}")
    else:
        print("검색 결과 없음 | context_used=False")

    # 6) 참고 컨텍스트/소스 출력
    if use_context:
        print("\n===== 참고된 청크 (점수순) =====")
        if meta_dbg is not None:
            # 멀티 인덱스 경로: meta_dbg에 retriever(A/B), source, score 포함
            for i, (md, p) in enumerate(zip(meta_dbg[:3], passages[:top_k]), start=1):
                tag = md.get("retriever", "?")
                src = md.get("source", "unknown")
                sc  = md.get("score", float("nan"))
                print(f"\n[{i}] [ce_score={sc:.3f}] retriever={tag} | source={src}\n{p[:400]}...")
        else:
            # 단일 인덱스 폴백 경로 (남겨둠)
            try:
                srcs = retrA.get_sources(hits)
                for (idx, score), p, s in zip(hits, passages, srcs):
                    print(f"\n[ce_score={score:.3f}] chunk#{idx} | source={s}\n{p[:400]}...")
            except Exception:
                for (idx, score), p in zip(hits, passages):
                    print(f"\n[score={score:.3f}] chunk#{idx}\n{p[:400]}...")


def cmd_run(args):
    import os
    import re
    import torch
    import pandas as pd

    # 1) 모델/인덱스 로드
    pipe = load_model()
    retrA = RAGRetriever(index_dir=os.path.join(INDEX_DIR, "laws"), device="cpu")
    retrB = RAGRetriever(index_dir=os.path.join(INDEX_DIR, "supplement"), device="cpu")

    # 2) Reranker
    rr_model = getattr(args, "rerank_model", "/workspace/models/gte-multilingual-reranker-base")
    rr_device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = STReranker(model_name_or_path=rr_model, device=rr_device, max_length=800)

    # 3) CSV 로드
    df = pd.read_csv(args.csv)

    preds, context_flags, full_context = [], [], []
    generated_texts, top_sources, top_scores = [], [], []

    top_k = getattr(args, "top_k", TOP_K)
    score_threshold = getattr(args, "threshold", SCORE_THRESHOLD)
    rerank_threshold = getattr(args, "rerank_threshold", RERANK_THRESHOLD)
    M_generic = getattr(args, "M_generic", M_GENERIC)
    M_filtered = getattr(args, "M_filtered", M_FILTERED)
    M_generic_B = getattr(args, "M_generic_B", M_GENERIC)   # ★ supplement 기본 후보 수
    keep_for_ce = getattr(args, "keep_for_ce", 50)
    use_clause = getattr(args, "use_clause", True)

    from tqdm import tqdm
    for idx, q in enumerate(tqdm(df['Question'], desc="Inference")):

        prompt, gen, passages, hits, use_context, contexts, meta_dbg = answer_with_rag(
            q,
            retrieverA=retrA,
            retrieverB=retrB,
            pipe=pipe,
            top_k=top_k,
            score_threshold=score_threshold,
            reranker=reranker,
            rerank_threshold=rerank_threshold,
            M_generic=M_generic,
            M_filtered=M_filtered,
            M_generic_B=M_generic_B,
            keep_for_ce=keep_for_ce,
            use_clause=use_clause,
        )

        ans = extract_answer_only(gen, original_question=q, prompt=prompt)

        preds.append(ans)
        context_flags.append(bool(use_context))
        full_context.append(contexts)
        generated_texts.append(gen)

        # (옵션) 디버깅용 메타 저장
        if meta_dbg is not None:
            # 멀티 인덱스 경로: answer_with_rag에서 이미 source와 점수를 제공
            top_sources.append([(d["retriever"], d["source"]) for d in meta_dbg[:top_k]])
            top_scores.append([float(d["score"]) for d in meta_dbg[:top_k]])
        else:
            # 단일 인덱스 폴백 경로
            try:
                srcs = retrA.get_sources(hits)
                top_sources.append(srcs[:top_k])
                top_scores.append([float(s) for _, s in hits[:top_k]])
            except Exception:
                top_sources.append([])
                top_scores.append([])

    # 4) 제출 파일 저장
    experiment_name = "result.csv"
    print("📄 제출 파일 생성 중...")
    sample_submission = pd.read_csv("data/sample_submission.csv")
    sample_submission['Answer'] = preds

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    sample_submission.to_csv(OUTPUT_PATH + experiment_name, index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {OUTPUT_PATH + experiment_name}")

    # 5) 부가 정보 저장
    result_with_info = sample_submission.copy()
    result_with_info["ContextUsed"] = context_flags
    result_with_info["Contexts"] = full_context
    result_with_info["Generated"] = generated_texts
    result_with_info["TopSources"] = top_sources
    result_with_info["TopScores"] = top_scores

    result_with_info_path = os.path.join(OUTPUT_PATH, "result_with_info.csv")
    result_with_info.to_csv(result_with_info_path, index=False, encoding='utf-8-sig')
    print(f"✅ 부가 정보 파일 저장 완료: {result_with_info_path}")



def main():
    parser = argparse.ArgumentParser(description="RAG for 전자금융거래법")
    sub = parser.add_subparsers()

    p_build = sub.add_parser("build", help="PDF에서 인덱스 생성(여러 개 가능)")
    p_build.add_argument("--pdf", nargs="+", help="PDF 파일 경로(공백으로 여러 개)")
    p_build.add_argument("--dir", help="PDF 폴더 경로(내부 *.pdf 일괄)")
    p_build.set_defaults(func=cmd_build)

    p_ask = sub.add_parser("ask", help="단일 질문에 RAG 적용")
    p_ask.add_argument("--question", required=True, help="질문 텍스트")
    p_ask.add_argument("--top_k", type=int, default=TOP_K)
    p_ask.add_argument("--threshold", type=float, default=SCORE_THRESHOLD, help="컨텍스트 사용 점수 임계값")
    p_ask.set_defaults(func=cmd_ask)

    p_run = sub.add_parser("run", help="CSV(Question 컬럼) 일괄 추론")
    p_run.add_argument("--csv", required=True, help="CSV 경로 (Question 컬럼 필요)")
    p_run.add_argument("--top_k", type=int, default=TOP_K)
    p_run.add_argument("--verbose", action="store_true")
    p_run.add_argument("--threshold", type=float, default=SCORE_THRESHOLD, help="컨텍스트 사용 점수 임계값")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
