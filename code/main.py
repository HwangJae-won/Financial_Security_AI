# main.py 파일 전체 내용

import sys
import os
import pandas as pd
from tqdm import tqdm

from rag import setup_retriever, answer_with_rag, rerank_documents
from model import load_llm_and_tokenizer, load_reranker_model

from config import (
    EMBEDDING_MODEL_NAME, DATA_PATH, LAW_PATH, TOP_K, SCORE_THRESHOLD,
    MODEL_NAME, CACHE_DIR, LOCAL_DIR_EXAONE
)

os.environ['HUGGINGFACE_HUB_CACHE'] = '/dev/shm/huggingface_cache'

def main_rerank():
    print("--- Financial Security AI Rerank Model Started---")
    
    # Reranker 모델 로드 (model.py에 구현)
    print("✨ Reranker 모델 로딩 중: sbert.net/ms-marco-TinyBERT-L-2-v2")
    reranker_model = load_reranker_model()

    # 기존 RAG 파이프라인
    retriever = setup_retriever(folder_path=LAW_PATH)
    llm, tokenizer = load_llm_and_tokenizer(MODEL_NAME, CACHE_DIR)
    print("--- ✅ RAG 및 LLM 초기 설정 완료 ---\n")

    print("--- 테스트 데이터 로딩 시작 ---")
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    print(f"✅ 테스트 데이터 로드 완료, 총 문항 수: {len(test_df)}\n")
    print("--- Rerank 기반 추론 시작 ---")

    preds = []
    
    # 리랭크는 넓게 검색하고 좁게 필터링하는 전략을 사용합니다.
    WIDE_TOP_K = 50 # 1차 검색에서 가져올 문서 수
    FINAL_TOP_K = 5 # 리랭킹 후 LLM에 전달할 문서 수

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Rerank 추론 진행"):
        question = row['Question']

        # 1. 1차 검색 (넓게 가져오기)
        print(f"🔍 1차 검색: {WIDE_TOP_K}개 문서 가져오기...")
        retrieved_docs_with_scores = retriever.similarity_search_with_score(
            question, 
            k=WIDE_TOP_K
        )
        retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]
        
        # 2. 재순위화 모듈을 사용하여 문서의 순위를 재조정합니다.
        print(f"🔄 문서 재순위화 중...")
        reranked_docs = rerank_documents(
            query=question, 
            documents=retrieved_docs, 
            reranker_model=reranker_model, 
            k=FINAL_TOP_K
        )

        # 3. 재순위화된 문서들을 컨텍스트로 사용하여 답변 생성
        final_answer = answer_with_rag(
            question=question, 
            vectorstore=retriever, # answer_with_rag 함수는 vectorstore 인자를 필요로 하므로 전달
            model=llm, 
            tokenizer=tokenizer,
            top_k=FINAL_TOP_K, # reranking 이후의 최종 top_k 전달
            reranked_docs=reranked_docs
        )
        
        preds.append(final_answer)

    print("--- ✅ Rerank 기반 추론 완료 ---\n")

    print("--- 제출 파일 생성 시작 ---")
    OUTPUT_PATH = "results/"
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    submission_df = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
    submission_df['Answer'] = preds
    submission_df.to_csv(os.path.join(OUTPUT_PATH, "exaone_rerank_v2.csv"), index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {os.path.join(OUTPUT_PATH, 'exaone_rerank_v2.csv')}")
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
    submission_df.to_csv(os.path.join(OUTPUT_PATH, "rerank_add.csv"), index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {os.path.join(OUTPUT_PATH, 'rerank_add.csv')}")
    print("--- ✅ 모든 작업 완료 ---")


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # main()
    main_rerank()