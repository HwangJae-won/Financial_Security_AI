# main.py 파일 전체 내용

import sys
import os
import pandas as pd
from tqdm import tqdm

from rag import setup_retriever, answer_with_rag
from model import load_llm_and_tokenizer

from config import (
    EMBEDDING_MODEL_NAME, DATA_PATH, LAW_PATH, TOP_K, SCORE_THRESHOLD,
    MODEL_NAME, CACHE_DIR, LOCAL_DIR_EXAONE
)

os.environ['HUGGINGFACE_HUB_CACHE'] = '/dev/shm/huggingface_cache'

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
    submission_df.to_csv(os.path.join(OUTPUT_PATH, "exaone_final.csv"), index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {os.path.join(OUTPUT_PATH, 'exaone_final.csv')}")
    print("--- ✅ 모든 작업 완료 ---")


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()