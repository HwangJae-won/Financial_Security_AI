import os
import re
from typing import List, Optional

from sentence_transformers import CrossEncoder

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma as LangchainChroma
from langchain_huggingface import HuggingFaceEmbeddings

from text_utils import load_pdf_text, chunk_law_text, _clean_text
from utils import extract_answer_only, is_multiple_choice, clean_markdown
from prompt import make_prompt_rag_exaone
from utils import load_and_chunk_file, load_all_documents, extract_metadata_from_filename

from config import EMBEDDING_MODEL_NAME, LAW_PATH, CHROMA_PERSIST_DIRECTORY

# -----------------------------
# 검색/리랭커 기본 설정 (config.py에서 분리)
# -----------------------------
TOP_K = 1
SCORE_THRESHOLD = 0.89

TOP_K_MC_DEFAULT=15
SCORE_THRESHOLD_MC_DEFAULT = 0.85
TOP_K_SUB_DEFAULT = 30
SCORE_THRESHOLD_SUB_DEFAULT = 0.75
def _rrf(rank: int, k: int = 60): return 1.0 / (k + rank)

def _fuse_rrf(cands_lists, keep_for_ce=50):
    merged = {}
    for cands in cands_lists:
        for rank, c in enumerate(cands, 1):
            key = c["source"]
            merged.setdefault(key, {**c, "rrf":0.0})
            merged[key]["rrf"] += _rrf(rank)
    return sorted(merged.values(), key=lambda x: x["rrf"], reverse=True)[:keep_for_ce]

def _search_global(vs, q, k, tag):
    res = vs.similarity_search_with_score(q, k=k)
    return [{"text":d.page_content, "score":float(s),
             "source":d.metadata.get("source","unknown"), "retriever":tag} for d,s in res]

def _search_filtered_by_law(vs, q, law_name, k, tag):
    if not law_name: return []
    res = vs.similarity_search_with_score(q, k=k, filter={"law_name": law_name})
    if not res:
        res = vs.similarity_search_with_score(q, k=k, filter={"law": law_name})
    return [{"text":d.page_content, "score":float(s),
             "source":d.metadata.get("source","unknown"), "retriever":tag} for d,s in res]

def _rerank(query, candidates, reranker, topn=1, batch_size=32):
    if not candidates: return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs, batch_size=batch_size, convert_to_numpy=True)
    for c, s in zip(candidates, scores): c["ce_score"] = float(s)
    return sorted(candidates, key=lambda x: x["ce_score"], reverse=True)[:topn]

def answer_with_rag_multi(question, retrieverA, retrieverB, model, tokenizer,
                          top_k=1, score_threshold=0.75,
                          reranker=None, rerank_threshold=0.0,
                          M_generic_A=20, M_filtered_A=10, M_generic_B=20, keep_for_ce=50):
    is_mc, _ = is_multiple_choice(question)
    law_hint = _extract_law_name_from_question(question)  # 기존 함수 재사용

    cand_gA = _search_global(retrieverA, question, M_generic_A, "A")
    cand_fA = _search_filtered_by_law(retrieverA, question, law_hint, M_filtered_A, "A")
    cand_gB = _search_global(retrieverB, question, M_generic_B, "B")

    fused = _fuse_rrf([cand_fA, cand_gA, cand_gB], keep_for_ce=keep_for_ce)

    if reranker and fused:
        ranked = _rerank(question, fused, reranker, topn=max(1, top_k))
        top_score = ranked[0]["ce_score"]
        gate_ok = (top_score >= (rerank_threshold if is_mc else score_threshold))
    else:
        ranked = sorted(fused, key=lambda x: x["score"], reverse=True)[:max(1, top_k)]
        top_score = ranked[0]["score"] if ranked else float("-inf")
        gate_ok = (top_score >= score_threshold)

    contexts = [c["text"] for c in ranked[:top_k]] if gate_ok else []
    prompt = make_prompt_rag_exaone(text=question, contexts=contexts, use_fewshot=True)

    # 1차 greedy
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=(2 if is_mc else 256), do_sample=False)
    gen = tokenizer.decode(out[0], skip_special_tokens=True)
    ans = extract_answer_only(clean_markdown(gen), original_question=question, prompt=prompt)
    if ans not in ("0","미응답"): return ans

    # 2차 샘플링
    outs = model.generate(**inputs, max_new_tokens=(2 if is_mc else 256),
                          do_sample=True, temperature=0.6, top_p=0.95,
                          num_return_sequences=3, repetition_penalty=1.05)
    for o in outs:
        cand = tokenizer.decode(o, skip_special_tokens=True)
        a = extract_answer_only(cand, original_question=question, prompt=prompt)
        if a not in ("0","미응답"): return a

    if is_mc:
        out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.6, top_p=0.95)
        gen2 = tokenizer.decode(out[0], skip_special_tokens=True)
        return extract_answer_only(gen2, original_question=question, prompt=prompt)
    return "미응답"
def rerank_documents(query: str, documents: List[Document], reranker_model: CrossEncoder, k: int) -> List[Document]:
    """
    쿼리와 문서들을 리랭킹 모델로 재순위화하고 상위 k개 문서를 반환합니다.
    """
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker_model.predict(pairs)
    
    doc_with_scores = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
    return [doc for score, doc in doc_with_scores[:k]]


def _extract_law_name_from_question(question: str) -> Optional[str]:
    """
    질문 텍스트에서 알려진 법률명을 추출합니다.
    """
    law_names = [
    "정보통신망 이용촉진 및 정보보호 등에 관한 법률 시행령",
    "정보통신망 이용촉진 및 정보보호 등에 관한 법률 시행규칙",
    "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
    "신용정보의 이용 및 보호에 관한 법률 시행규칙",
    "신용정보의 이용 및 보호에 관한 법률 시행령",
    "신용정보의 이용 및 보호에 관한 법률",
    "신용정보보안감독규정",
    "전자금융거래법 시행령",
    "전자금융거래법",
    "전자금융감독규정",
    "전자서명법 시행규칙",
    "전자서명법 시행령",
    "전자서명법",
    "개인정보보호법 시행령",
    "개인정보보호법"]
    for name in law_names:
        if name in question:
            return name
    return None
def answer_with_rag(
    question: str,
    vectorstore,
    model,    
    tokenizer,
    top_k: int = TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    reranked_docs: Optional[List[Document]] = None
):
    """
    RAG 기반으로 질문에 답하는 다단계 추론 함수.
    """
    # --- 1. 질문 유형에 따라 파라미터를 동적으로 설정 (가장 먼저 수행) ---
    is_mc, _ = is_multiple_choice(question)
    
    current_top_k = top_k
    current_score_threshold = score_threshold
    
    if is_mc:
        current_top_k = TOP_K_MC_DEFAULT
        current_score_threshold = SCORE_THRESHOLD_MC_DEFAULT
        print(f"📄 객관식 문제 감지: TOP_K={current_top_k}, SCORE_THRESHOLD={current_score_threshold} 적용")
    else:
        current_top_k = TOP_K_SUB_DEFAULT
        current_score_threshold = SCORE_THRESHOLD_SUB_DEFAULT
        print(f"📄 주관식 문제 감지: TOP_K={current_top_k}, SCORE_THRESHOLD={current_score_threshold} 적용")
        print(f"🔍 질문: {question}") # 주관식 질문만 출력
    
    # --- 2. 동적으로 결정된 파라미터로 검색을 단 한 번만 수행 ---
    if reranked_docs is not None:
        if not is_mc:
            print("✅ 리랭킹된 문서 사용.")
        documents = reranked_docs
        contexts = [doc.page_content for doc in documents]
    else:
        law_name_from_question = _extract_law_name_from_question(question)
        if law_name_from_question:
            if not is_mc:
                print(f"🎯 특정 법률명 감지: {law_name_from_question}. 해당 법률 문서만 검색합니다.")
            docs_with_scores = vectorstore.similarity_search_with_score(
                question, 
                k=current_top_k, 
                filter={"law_name": law_name_from_question}
            )
        else:
            docs_with_scores = vectorstore.similarity_search_with_score(question, k=current_top_k)

        documents = [doc for doc, score in docs_with_scores]
        scores = [score for doc, score in docs_with_scores]
        contexts = [doc.page_content for doc in documents]
        
        # ★ 검색 결과 및 출처 로깅 (주관식일 경우) ★
        if not is_mc:
            print("\n[검색 결과]")
            if not documents:
                print("❗ 검색된 문서가 없습니다.")
            for i, (doc, score) in enumerate(zip(documents, scores)):
                source_info = doc.metadata.get('source', '출처 정보 없음')
                print(f"  - {i+1}위 (점수: {score:.4f}): '{doc.page_content[:50]}...' [출처: {source_info}]")
        
        # ★ 수정된 부분: 동적으로 설정된 current_score_threshold를 사용합니다.
        if not scores or scores[0] < current_score_threshold:
            if not is_mc:
                print(f"❌ 최고 점수({scores[0]:.4f})가 임계값({current_score_threshold:.4f}) 미만이므로 컨텍스트를 비웁니다.")
            contexts = []
        else:
            if not is_mc:
                print(f"\n[프롬프트에 사용될 최종 컨텍스트 수]: {len(contexts)}")
                for i, context in enumerate(contexts):
                    print(f"  - 컨텍스트 {i+1}: '{context[:100]}...'")
            
    # --- 3. 프롬프트 생성 ---
    prompt = make_prompt_rag_exaone(
        text=question,
        contexts=contexts,
        use_fewshot=True
    )
    # ★ 생성된 전체 프롬프트 로깅 (주관식일 경우) ★
    if not is_mc:
        print("\n[생성된 전체 프롬프트]")
        print("--------------------")
        print(prompt)
        print("--------------------")

    
    def generate_answer(prompt, **gen_params):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, **gen_params)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # --- 4. 다단계 추론 (단일 결과 변수 사용) ---
    final_answer = "미응답"

    if not is_mc:
        print("🚀 1차 추론 시작 (그리디)...")
    gen_text = generate_answer(
        prompt, 
        max_new_tokens=2 if is_mc else 256,
        do_sample=False,
    )
    # ★ 1차 추론 결과 로깅 (주관식일 경우) ★
    if not is_mc:
        print("\n[1차 추론 결과 (원본)]")
        print(gen_text)
    temp_answer = extract_answer_only(generated_text=clean_markdown(gen_text), 
                                      original_question=question, prompt=prompt)
    if not is_mc:
        print("\n[1차 추론 결과 (후처리)]")
        print(temp_answer)

    if temp_answer not in ("0", "미응답"):
        final_answer = temp_answer
    
    if final_answer in ("0", "미응답"):
        if not is_mc:
            print("🔁 1차 실패, 2차 추론 시작 (샘플링)...")
        gen_params_retry = {
            "max_new_tokens": 2 if is_mc else 256,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "num_return_sequences": 3,
            "repetition_penalty": 1.05
            
        }
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs_retry = model.generate(**inputs, **gen_params_retry)
        
        for o in outputs_retry:
            cand_text = tokenizer.decode(o, skip_special_tokens=True)
            # ★ 2차 추론 결과 로깅 (주관식일 경우) ★
            if not is_mc:
                print(f"\n[2차 추론 후보 (원본)]")
                print(cand_text)
            
            cand_answer = extract_answer_only(cand_text, original_question=question, prompt=prompt)
            if not is_mc:
                print(f"[2차 추론 후보 (후처리)]")
                print(cand_answer)
            
            if cand_answer not in ("0", "미응답"):
                final_answer = cand_answer
                if not is_mc:
                    print("✅ 2차 추론 성공")
                break
    
    if final_answer in ("0", "미응답") and is_mc:
        print("❌ 최종 실패, 객관식 후처리 시도...")
        gen_params_fallback = {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95
        }
        final_gen_text = generate_answer(prompt, **gen_params_fallback)
        
        # ★ 수정된 부분 ★
        # extract_answer_only를 먼저 적용하여 텍스트를 정리
        temp_answer = extract_answer_only(final_gen_text, original_question=question, prompt=prompt)
        
        # 그 결과물에 대해 숫자 정규식 적용
        m = re.search(r"\b([1-9][0-9]?)\b", temp_answer)
        if m:
            final_answer = m.group(1)
            print("✅ 후처리 성공")

    if final_answer in ("0", "미응답"):
        if not is_mc:
            print("❌ 모든 시도 실패 (미응답)")
    
    print("Final Answer:", final_answer)
    return final_answer

def setup_retriever(folder_path="laws/", persist_dir=None):
    print(f"📄 '{folder_path}' 폴더에서 문서를 로딩하고 벡터화합니다.")
    if persist_dir is None:
        persist_dir = CHROMA_PERSIST_DIRECTORY
    if not os.path.exists(persist_dir):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"'{folder_path}' 폴더를 찾을 수 없습니다.")
        docs = load_all_documents(folder_path)
        print(f"✅ 분할된 문서 수: {len(docs)}")
        if not docs:
            raise ValueError("문서 청크가 생성되지 않았습니다.")
        print(f"✨ 임베딩 모델 로딩 중: {EMBEDDING_MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cuda"}, encode_kwargs={"normalize_embeddings": True})
        print("🔍 벡터 스토어(ChromaDB) 생성 중...")
        vectorstore = LangchainChroma.from_documents(
            docs, embeddings, persist_directory=persist_dir
        )
        print("✅ 벡터 스토어 완료.")
    else:
        print(f"✨ 임베딩 모델 로딩 중: {EMBEDDING_MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cuda"}, encode_kwargs={"normalize_embeddings": True})
        
        vectorstore = LangchainChroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        
        # ChromaDB에 저장된 문서 수를 확인합니다.
        try:
            doc_count = vectorstore._collection.count()
            
            if doc_count == 0:
                print("⚠️ 경고: 로드된 ChromaDB 인덱스가 비어 있습니다. 인덱스를 다시 생성합니다.")
                # 비어 있는 인덱스 폴더를 삭제하고 함수를 재귀적으로 호출하여 재생성
                import shutil
                shutil.rmtree(persist_dir)
                return setup_retriever(folder_path, persist_dir=persist_dir)
            
            print(f"✅ ChromaDB 인덱스 로드 완료. 문서 수: {doc_count}")
        except Exception as e:
            print(f"❌ 오류: ChromaDB 로드 중 오류 발생 ({e}). 인덱스가 손상된 것으로 보입니다. 다시 생성합니다.")
            import shutil
            shutil.rmtree(persist_dir)
            return setup_retriever(folder_path, persist_dir=persist_dir)
        
    return vectorstore