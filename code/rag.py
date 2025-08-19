import os
import re
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from transformers import pipeline
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma as LangchainChroma
import chromadb
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# text_utils와 utils 모듈에서 필요한 함수들을 import
from text_utils import load_pdf_text, chunk_law_text, _clean_text
from utils import extract_answer_only, is_multiple_choice, clean_markdown
from prompt import make_prompt_rag_exaone

# config 모듈에서 변수 import (필요에 따라 수정)
from config import EMBEDDING_MODEL_NAME, SCORE_THRESHOLD, TOP_K, LAW_PATH, CHROMA_PERSIST_DIRECTORY

TOP_K_MC_DEFAULT=15
SCORE_THRESHOLD_MC_DEFAULT = 0.85
TOP_K_SUB_DEFAULT = 30
SCORE_THRESHOLD_SUB_DEFAULT = 0.75

def rerank_documents(query: str, documents: List[Document], reranker_model: CrossEncoder, k: int) -> List[Document]:
    """
    쿼리와 문서들을 리랭킹 모델로 재순위화하고 상위 k개 문서를 반환합니다.
    """
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker_model.predict(pairs)
    
    doc_with_scores = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
    return [doc for score, doc in doc_with_scores[:k]]

def _extract_metadata_from_filename(filename: str) -> Dict[str, Optional[str]]:
    """
    다양한 파일명 패턴에서 법률 메타데이터를 추출합니다.
    (이 함수는 setup_retriever에서만 사용됩니다.)
    """
    # 1. 가장 흔한 패턴: (유형)(제...호)(날짜)
    pattern1 = re.compile(r'(.+?)\((.+?)\)\(제(.+?호)\)\((\d+)\)')
    match1 = pattern1.search(filename)
    if match1:
        return {
            "law_name": match1.group(1).strip(),
            "law_type": match1.group(2),
            "law_number": match1.group(3),
            "enactment_date": match1.group(4)
        }

    # 2. 번호가 없는 경우를 대비한 패턴: (유형).pdf
    pattern2 = re.compile(r'(.+?)\((.+?)\)\.pdf')
    match2 = pattern2.search(filename)
    if match2:
        return {
            "law_name": match2.group(1).strip(),
            "law_type": match2.group(2),
            "law_number": None,
            "enactment_date": None
        }

    # 3. 그 외 알 수 없는 형식을 대비한 최후의 패턴
    pattern3 = re.compile(r'(.+?)\.pdf')
    match3 = pattern3.search(filename)
    if match3:
        return {
            "law_name": match3.group(1).strip(),
            "law_type": None,
            "law_number": None,
            "enactment_date": None
        }
    
    print(f"⚠️ 경고: 파일명 패턴 불일치 - {filename}")
    return {}

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

    # --- 2. 동적으로 결정된 파라미터로 검색을 단 한 번만 수행 ---
    if reranked_docs is not None:
        print("✅ 리랭킹된 문서 사용.")
        documents = reranked_docs
        contexts = [doc.page_content for doc in documents]
    else:
        law_name_from_question = _extract_law_name_from_question(question)
        if law_name_from_question:
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
        
        # ★ 수정된 부분: 동적으로 설정된 current_score_threshold를 사용합니다.
        if not scores or scores[0] < current_score_threshold:
            contexts = []
            
    # --- 3. 프롬프트 생성 ---
    prompt = make_prompt_rag_exaone(
        text=question,
        contexts=contexts,
        use_fewshot=True
    )
    
    def generate_answer(prompt, **gen_params):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, **gen_params)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # --- 4. 다단계 추론 (단일 결과 변수 사용) ---
    final_answer = "미응답"

    print("🚀 1차 추론 시작 (그리디)...")
    gen_text = generate_answer(
        prompt, 
        max_new_tokens=2 if is_mc else 256,
        do_sample=False,
    )
    temp_answer = extract_answer_only(generated_text=clean_markdown(gen_text), 
                                      original_question=question, prompt=prompt)
    if temp_answer not in ("0", "미응답"):
        final_answer = temp_answer
    
    if final_answer in ("0", "미응답"):
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
            cand_answer = extract_answer_only(cand_text, original_question=question, prompt=prompt)
            if cand_answer not in ("0", "미응답"):
                final_answer = cand_answer
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
        print("❌ 모든 시도 실패 (미응답)")
    print("Final Answer:", final_answer)
    return final_answer

def setup_retriever(folder_path="laws/"):
    print(f"📄 '{folder_path}' 폴더에서 문서를 로딩하고 벡터화합니다.")
    all_documents = []

    if not os.path.exists(CHROMA_PERSIST_DIRECTORY):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"'{folder_path}' 폴더를 찾을 수 없습니다.")

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension in [".pdf", ".txt"]:
                print(f"📄 로딩 중: {file_path}")
                
                if file_extension == ".pdf":
                    raw_text = load_pdf_text(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw_text = f.read()

                if not raw_text.strip():
                    print(f"⚠️ 경고: '{file_path}'에서 텍스트를 추출하지 못했습니다.")
                    continue
                
                chunks, labels = chunk_law_text(_clean_text(raw_text)) 
                file_metadata = _extract_metadata_from_filename(filename)

                for chunk, label in zip(chunks, labels):
                    combined_metadata = {
                        "source": file_path,
                        "label": label,
                        **file_metadata
                    }
                    all_documents.append(Document(page_content=chunk, metadata=combined_metadata))

            else:
                print(f"⚠️ 경고: 지원되지 않는 파일 형식 스킵 - {file_path}")
        
        docs = all_documents
        print(f"✅ 분할된 문서 수: {len(docs)}")
        if not docs:
            raise ValueError("문서 청크가 생성되지 않았습니다.")
        
        print(f"✨ 임베딩 모델 로딩 중: {EMBEDDING_MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        print("🔍 벡터 스토어(ChromaDB) 생성 중...")
        vectorstore = LangchainChroma.from_documents(
            docs, 
            embeddings, 
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        print("✅ 벡터 스토어 완료.")
        
    else:
        print(f"✨ 임베딩 모델 로딩 중: {EMBEDDING_MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = LangchainChroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        print(f"✅ ChromaDB 인덱스 로드 완료: {CHROMA_PERSIST_DIRECTORY}")
        
    return vectorstore

# def answer_with_rag(
#     question: str,
#     vectorstore,
#     model,    
#     tokenizer,
#     top_k: int = TOP_K,
#     score_threshold: float = SCORE_THRESHOLD
# ):
#     """
#     RAG 기반으로 질문에 답하는 다단계 추론 함수.
#     """
#     # --- 1. 질문 유형에 따라 파라미터를 동적으로 설정 (가장 먼저 수행) ---
#     is_mc, _ = is_multiple_choice(question)
    
#     current_top_k = top_k
#     current_score_threshold = score_threshold
    
#     if is_mc:
#         current_top_k = TOP_K_MC if 'TOP_K_MC' in locals() else TOP_K_MC_DEFAULT
#         current_score_threshold = SCORE_THRESHOLD_MC if 'SCORE_THRESHOLD_MC' in locals() else SCORE_THRESHOLD_MC_DEFAULT
#         print(f"📄 객관식 문제 감지: TOP_K={current_top_k}, SCORE_THRESHOLD={current_score_threshold} 적용")
#     else:
#         current_top_k = TOP_K_SUB if 'TOP_K_SUB' in locals() else TOP_K_SUB_DEFAULT
#         current_score_threshold = SCORE_THRESHOLD_SUB if 'SCORE_THRESHOLD_SUB' in locals() else SCORE_THRESHOLD_SUB_DEFAULT
#         print(f"📄 주관식 문제 감지: TOP_K={current_top_k}, SCORE_THRESHOLD={current_score_threshold} 적용")

#     # --- 2. 동적으로 결정된 파라미터로 검색을 단 한 번만 수행 ---
#     law_name_from_question = _extract_law_name_from_question(question)
#     if law_name_from_question:
#         print(f"🔍 '{law_name_from_question}' 필터를 적용하여 검색합니다.")
#         docs_with_scores = vectorstore.similarity_search_with_score(
#             question, 
#             k=current_top_k, 
#             filter={"law_name": law_name_from_question}
#         )
#     else:
#         print(f"🔍 필터 없이 전체 벡터 스토어를 대상으로 검색합니다.")
#         docs_with_scores = vectorstore.similarity_search_with_score(question, k=current_top_k)

#     documents = [doc for doc, score in docs_with_scores]
#     scores = [score for doc, score in docs_with_scores]
#     contexts = [doc.page_content for doc in documents]
    
#     if not scores or scores[0] < current_score_threshold:
#         contexts = []
    
#     # --- 3. 프롬프트 생성 ---
#     prompt = make_prompt_rag_exaone(
#         text=question,
#         contexts=contexts,
#         use_fewshot=True
#     )
    
#     def generate_answer(prompt, **gen_params):
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#         output = model.generate(**inputs, **gen_params)
#         return tokenizer.decode(output[0], skip_special_tokens=True)

#     # --- 4. 다단계 추론 (단일 결과 변수 사용) ---
#     final_answer = "미응답"

#     # 1차 시도 (그리디)
#     print("🚀 1차 추론 시작 (그리디)...")
#     gen_text = generate_answer(
#         prompt, 
#         max_new_tokens=2 if is_mc else 256,
#         do_sample=False,
#     )
#     temp_answer = extract_answer_only(gen_text, original_question=question, prompt=prompt)
#     if temp_answer not in ("0", "미응답"):
#         final_answer = temp_answer
    
#     # 2차 시도 (샘플링)
#     if final_answer in ("0", "미응답"):
#         print("🔁 1차 실패, 2차 추론 시작 (샘플링)...")
#         gen_params_retry = {
#             "max_new_tokens": 2 if is_mc else 256,
#             "do_sample": True,
#             "temperature": 0.6,
#             "top_p": 0.95,
#             "num_return_sequences": 3,
#             "repetition_penalty": 1.05
#         }
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#         outputs_retry = model.generate(**inputs, **gen_params_retry)
        
#         for o in outputs_retry:
#             cand_text = tokenizer.decode(o, skip_special_tokens=True)
#             cand_answer = extract_answer_only(cand_text, original_question=question, prompt=prompt)
#             if cand_answer not in ("0", "미응답"):
#                 final_answer = cand_answer
#                 print("✅ 2차 추론 성공")
#                 break
    
#     # 3차 시도 (객관식 전용 후처리)
#     if final_answer in ("0", "미응답") and is_mc:
#         print("❌ 최종 실패, 객관식 후처리 시도...")
#         gen_params_fallback = {
#             "max_new_tokens": 128,
#             "do_sample": True,
#             "temperature": 0.6,
#             "top_p": 0.95
#         }
#         final_gen_text = generate_answer(prompt, **gen_params_fallback)
#         m = re.search(r"\b([1-9][0-9]?)\b", final_gen_text)
#         if m:
#             final_answer = m.group(1)
#             print("✅ 후처리 성공")

#     if final_answer in ("0", "미응답"):
#         print("❌ 모든 시도 실패 (미응답)")
#     print("Final Answer:", final_answer)
#     return final_answer
