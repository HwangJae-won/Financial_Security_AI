import sys
import os
import re
import pandas as pd
from tqdm import tqdm

from rag import setup_retriever, answer_with_rag, rerank_documents, answer_with_rag_multi, TOP_K, SCORE_THRESHOLD
from model import load_llm_and_tokenizer, load_reranker_model

from config import (
    EMBEDDING_MODEL_NAME, DATA_PATH, LAW_PATH,
    MODEL_NAME, CACHE_DIR, LOCAL_DIR_EXAONE,
    CHROMA_LAWS_DIR, CHROMA_SUPP_DIR, SUPP_PATH
)

os.environ['HUGGINGFACE_HUB_CACHE'] = '/dev/shm/huggingface_cache'

def main_rerank():
    print("--- Financial Security AI Rerank Model Started---")
    
    # Reranker 모델 로드 (model.py에 구현)
    print("✨ Reranker 모델 로딩 중")
    reranker_model = load_reranker_model()

    # 기존 RAG 파이프라인 (멀티 인덱스)
    retrieverA = setup_retriever(folder_path=LAW_PATH, persist_dir=CHROMA_LAWS_DIR)
    if os.path.isdir(SUPP_PATH) and any(name.lower().endswith((".pdf", ".txt")) for name in os.listdir(SUPP_PATH)):
        retrieverB = setup_retriever(folder_path=SUPP_PATH, persist_dir=CHROMA_SUPP_DIR)
    else:
        print("⚠️ supplement 말뭉치 없음 → laws 인덱스를 B로 재사용")
        retrieverB = retrieverA

    llm, tokenizer = load_llm_and_tokenizer(MODEL_NAME, CACHE_DIR)
    print("--- ✅ RAG 및 LLM 초기 설정 완료 ---\n")

    print("--- 테스트 데이터 로딩 시작 ---")
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    print(f"✅ 테스트 데이터 로드 완료, 총 문항 수: {len(test_df)}\n")
    print("--- Rerank 기반 추론 시작 ---")

    preds = []

    TOP_K_FINAL = 5
    SCORE_THRESHOLD_SUB = 0.75
    RERANK_THRESHOLD_MC = 0.0
    M_GENERIC_A = 20
    M_FILTERED_A = 10
    M_GENERIC_B = 20
    KEEP_FOR_CE = 50

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Rerank 추론 진행"):
        question = row['Question']

        final_answer = answer_with_rag_multi(
            question=question,
            retrieverA=retrieverA,
            retrieverB=retrieverB,
            model=llm,
            tokenizer=tokenizer,
            top_k=TOP_K_FINAL,
            score_threshold=SCORE_THRESHOLD_SUB,
            reranker=reranker_model,
            rerank_threshold=RERANK_THRESHOLD_MC,
            M_generic_A=M_GENERIC_A,
            M_filtered_A=M_FILTERED_A,
            M_generic_B=M_GENERIC_B,
            keep_for_ce=KEEP_FOR_CE
        )
        
        preds.append(final_answer)

    print("--- ✅ Rerank 기반 추론 완료 ---\n")

    print("--- 제출 파일 생성 시작 ---")
    OUTPUT_PATH = "results/"
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    submission_df = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
    submission_df['Answer'] = preds
    submission_df.to_csv(os.path.join(OUTPUT_PATH, "exaone_rerank_v3.csv"), index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {os.path.join(OUTPUT_PATH, 'exaone_rerank_v3.csv')}")
    print("--- ✅ 모든 작업 완료 ---")
    
def main():
    print("--- Financial Security AI Model Started---")   
    
    vectorstore = setup_retriever(folder_path=LAW_PATH)
    model, tokenizer = load_llm_and_tokenizer(MODEL_NAME, CACHE_DIR)
    
    print("--- ✅ RAG 및 LLM 초기 설정 완료 ---\n")

    print("--- 테스트 데이터 로딩 시작 ---")
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    print(f"✅ 테스트 데이터 로드 완료, 총 문항 수: {len(test_df)}\n")
    print("--- RAG 기반 추론 시작 ---")

    preds = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="RAG 추론 진행"):
        question = row['Question']
        
        final_answer = answer_with_rag(
            question=question, 
            vectorstore=vectorstore, 
            model=model, 
            tokenizer=tokenizer,
            top_k=TOP_K,
            score_threshold=SCORE_THRESHOLD
        )
        
        preds.append(final_answer)

    print("--- ✅ RAG 기반 추론 완료 ---\n")
   
    print("--- 제출 파일 생성 시작 ---")
    OUTPUT_PATH = "results/"
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    submission_df = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
    submission_df['Answer'] = preds
    submission_df.to_csv(os.path.join(OUTPUT_PATH, "exaone_rerank_v4.csv"), index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {os.path.join(OUTPUT_PATH, 'exaone_rerank_v4.csv')}")
    print("--- ✅ 모든 작업 완료 ---")


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # main()
    main_rerank()