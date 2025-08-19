import sys
import os
import pandas as pd
from tqdm import tqdm
from rag import setup_retriever,answer_with_rag
from prompt import make_prompt_rag_exaone
from model import load_llm_and_tokenizer
from utils import extract_answer_only, is_multiple_choice

from config import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, DATA_PATH, LAW_PATH, TOP_K, SCORE_THRESHOLD, MODEL_NAME,CACHE_DIR,LOCAL_DIR_EXAONE
os.environ['HUGGINGFACE_HUB_CACHE'] = '/dev/shm/huggingface_cache'
def main():
    print("--- Financial Security AI Model Started---")   
    
    # 임베딩 및 벡터 저장소 설정 (setup_retriever 함수가 필요)
    vectorstore = setup_retriever(folder_path=LAW_PATH)
    
    # 모델과 토크나이저를 한 번만 로드합니다.
    model, tokenizer = load_llm_and_tokenizer(MODEL_NAME, CACHE_DIR)
    
    print("--- ✅ RAG 및 LLM 초기 설정 완료 ---\n")

    print("---  테스트 데이터 로딩 시작 ---")
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    print(f"✅ 테스트 데이터 로드 완료, 총 문항 수: {len(test_df)}\n")
    print("--- RAG 기반 추론 시작 ---")

    preds = []
    
    # answer_with_rag 함수에 model, tokenizer를 전달
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="RAG 추론 진행"):
        question = row['Question']
        
        # 기본 파라미터로 추론
        final_answer = answer_with_rag(
            question=question, 
            vectorstore=vectorstore, 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.05
        )

        if final_answer in ("0", "미응답"):
            # 재시도 시 파라미터만 변경하여 호출
            final_answer = answer_with_rag(
                question=question, 
                vectorstore=vectorstore, 
                model=model, 
                tokenizer=tokenizer,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.6, 
                top_p=0.95
            )

        if final_answer in ("0", "미응답"):
            # 2차 재시도
            final_answer = answer_with_rag(
                question=question, 
                vectorstore=vectorstore, 
                model=model, 
                tokenizer=tokenizer,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8, 
                top_p=1.0
            )
        
        preds.append(final_answer)

    print("--- ✅ RAG 기반 추론 완료 ---\n")
   

    print("--- 제출 파일 생성 시작 ---")
    OUTPUT_PATH = "results/"
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    submission_df = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
    submission_df['Answer'] = preds
    submission_df.to_csv(os.path.join(OUTPUT_PATH, "exaone_retry.csv"), index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {os.path.join(OUTPUT_PATH, 'exaone_retry.csv')}")
    print("--- ✅ 모든 작업 완료 ---")


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()   
    
    
"""
Rerank 모듈 실행 

   
def main_rerank():
    print("--- Financial Security AI Model Started---")   
    retriever = setup_retriever(folder_path=LAW_PATH)
    llm = setup_model()
    print("--- ✅ RAG 및 LLM 초기 설정 완료 ---\n")

    print("---  테스트 데이터 로딩 시작 ---")
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    print(f"✅ 테스트 데이터 로드 완료, 총 문항 수: {len(test_df)}\n")

    print("--- RAG 기반 추론 시작 ---")
    preds = []
    USE_FEWSHOT = True 
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="RAG 추론 진행"):
        question = row['Question']

        # 1차 검색 (더 많은 후보군을 얻기 위해 k값을 10~20으로 설정하는 것이 좋습니다.)
        retrieved_docs = retriever.invoke(question)

        # 2. 재순위화 모듈을 사용하여 문서의 순위를 재조정합니다.
        # 최종적으로 상위 5개의 문서만 사용하도록 k=5 설정
        reranked_docs = rerank_documents(question, retrieved_docs, k=5) 

        # 재순위화된 문서들을 컨텍스트로 사용
        contexts_list = [doc.page_content for doc in reranked_docs]

        prompt = make_prompt_rag_solar(
            text=question,
            contexts=contexts_list,
            use_fewshot=True
        )

        answer_text = llm.invoke(prompt)
        
        final_answer = extract_answer_only(
            generated_text=f"답변:{answer_text}",
            original_question=question,
            prompt=prompt 
        )
        preds.append(final_answer)

    print("--- ✅ RAG 기반 추론 완료 ---\n")

    print("--- 제출 파일 생성 시작 ---")
    OUTPUT_PATH = "results/"
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    submission_df = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
    submission_df['Answer'] = preds
    submission_df.to_csv(os.path.join(OUTPUT_PATH, "pdf_mudlue_change_processing.csv"), index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {os.path.join(OUTPUT_PATH, 'fit_condition_solar.csv')}")
    print("--- ✅ 모든 작업 완료 ---")

"""
