
import os
import re
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from transformers import pipeline
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma as LangchainChroma
import chromadb
import torch
import re
from langchain_huggingface import HuggingFaceEmbeddings
# text_utils와 utils 모듈에서 필요한 함수들을 import
from text_utils import load_pdf_text, chunk_law_text, _clean_text
from utils import extract_answer_only, is_multiple_choice
from prompt import make_prompt_rag_exaone

# config 모듈에서 변수 import (필요에 따라 수정)
from config import (
    EMBEDDING_MODEL_NAME, SCORE_THRESHOLD, TOP_K, LAW_PATH
)

# ChromaDB의 데이터 저장 경로 (config.py에 추가해야 함)
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

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
        "개인정보보호법", "신용정보 이용 및 보호에 관한 법률", "전자금융거래법",
        "전자서명법", "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
        "신용정보보안감독규정" # 예시
    ]
    for name in law_names:
        if name in question:
            return name
    return None
# rag.py 파일의 answer_with_rag 함수를 이 코드로 교체하세요.

def answer_with_rag(
    question: str,
    vectorstore,
    model,    
    tokenizer,
    top_k: int = 20,
    score_threshold: float = 0.89
):
    """
    RAG 기반으로 질문에 답하는 다단계 추론 함수.
    """
    # --- 1. 문서 검색 ---
    law_name_from_question = _extract_law_name_from_question(question)
    
    if law_name_from_question:
        print(f"🔍 '{law_name_from_question}' 필터를 적용하여 검색합니다.")
        docs_with_scores = vectorstore.similarity_search_with_score(
            question, 
            k=top_k, 
            filter={"law_name": law_name_from_question}
        )
    else:
        print(f"🔍 필터 없이 전체 벡터 스토어를 대상으로 검색합니다.")
        docs_with_scores = vectorstore.similarity_search_with_score(question, k=top_k)

    documents = [doc for doc, score in docs_with_scores]
    
    # ★ 수정된 부분 ★: 점수만 올바르게 추출합니다.
    scores = [score for doc, score in docs_with_scores]
    
    contexts = [doc.page_content for doc in documents]
    
    if not scores or scores[0] < score_threshold:
        contexts = []
    
    # --- 2. 프롬프트 생성 ---
    prompt = make_prompt_rag_exaone(
        text=question,
        contexts=contexts,
        use_fewshot=True
    )
    
    is_mc, _ = is_multiple_choice(question)
    
    def generate_answer(prompt, **gen_params):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, **gen_params)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # --- 3. 다단계 추론 (단일 결과 변수 사용) ---
    final_answer = "미응답"

    # 1차 시도 (그리디)
    print("🚀 1차 추론 시작 (그리디)...")
    gen_text = generate_answer(
        prompt, 
        max_new_tokens=2 if is_mc else 256,
        do_sample=False,
    )
    temp_answer = extract_answer_only(gen_text, original_question=question, prompt=prompt)
    if temp_answer not in ("0", "미응답"):
        final_answer = temp_answer
    
    # 2차 시도 (샘플링)
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
    
    # 3차 시도 (객관식 전용 후처리)
    if final_answer in ("0", "미응답") and is_mc:
        print("❌ 최종 실패, 객관식 후처리 시도...")
        gen_params_fallback = {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95
        }
        final_gen_text = generate_answer(prompt, **gen_params_fallback)
        m = re.search(r"\b([1-9][0-9]?)\b", final_gen_text)
        if m:
            final_answer = m.group(1)
            print("✅ 후처리 성공")

    if final_answer in ("0", "미응답"):
        print("❌ 모든 시도 실패 (미응답)")
    print("Final Answer:", final_answer)
    return final_answer

def setup_retriever(folder_path="laws/"):
    # setup_retriever 함수는 여기에 그대로 두시면 됩니다.
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