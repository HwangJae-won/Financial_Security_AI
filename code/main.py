import sys
import os
import pandas as pd
from tqdm import tqdm
from rag import setup_retriever
from model import setup_model
from utils import extract_answer_only, is_multiple_choice
from prompt import make_prompt_rag_solar
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, DATA_PATH, LAW_PATH

def main():
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
        retrieved_docs = retriever.invoke(question)
        
        # retrieved_docs에서 page_content만 추출하여 리스트로 만듭니다.
        contexts_list = [doc.page_content for doc in retrieved_docs]
        
        prompt = make_prompt_rag_solar(
            text=question,          # 첫 번째 인자로 질문 텍스트
            contexts=contexts_list, # 두 번째 인자로 컨텍스트 리스트
            use_fewshot=USE_FEWSHOT # 세 번째 인자로 Few-shot 사용 여부
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
    submission_df.to_csv(os.path.join(OUTPUT_PATH, "solar_more_large_model_rag.csv"), index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {os.path.join(OUTPUT_PATH, 'solar_more_large_model_rag.csv')}")
    print("--- ✅ 모든 작업 완료 ---")


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()