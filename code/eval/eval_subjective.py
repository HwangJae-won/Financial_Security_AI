import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Helper Functions ---

def get_openai_embedding(text: str) -> np.ndarray:
    """
    OpenAI API를 사용하여 텍스트의 임베딩 벡터를 생성합니다.
    """
    # YOUR_API_KEY를 실제 API 키로 교체하세요.
    client = OpenAI(api_key="") 
    
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)

def calculate_keyword_recall(predicted_text: str, keywords: list) -> float:
    """
    키워드 재현율 점수를 계산합니다.
    """
    if not isinstance(keywords, list) or not keywords:
        return 0.0
    
    predicted_text = str(predicted_text).lower()
    matched_keywords = [kw for kw in keywords if kw.lower() in predicted_text]
    return len(matched_keywords) / len(keywords)

def calculate_semantic_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    두 임베딩 벡터 간의 코사인 유사도를 계산합니다.
    """
    if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
        return 0.0
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

# --- 주관식 평가 함수 ---

def calculate_subjective_score(submission_file: str) -> float:
    """
    주관식 문제에 대한 모델 답변을 평가하고 평균 Mixed_Score를 반환하는 함수.
    """
    
    try:
        golden_df = pd.read_csv('data/test_subjective_answer.csv')
        golden_df.columns = golden_df.columns.str.strip()
        
        full_predictions_df = pd.read_csv(submission_file)
        
        subjective_ids = golden_df['ID'].tolist()
        predicted_df = full_predictions_df[full_predictions_df['ID'].isin(subjective_ids)].copy()
        predicted_df.rename(columns={'Answer': 'Predicted_Answer'}, inplace=True)

        merged_df = pd.merge(predicted_df, golden_df, on='ID', how='left')
        merged_df['Keywords'] = merged_df['Keywords'].str.replace(r'[()]', '', regex=True).str.split(', ')

        merged_df['Predicted_Embedding'] = merged_df['Predicted_Answer'].apply(get_openai_embedding)
        merged_df['Correct_Embedding'] = merged_df['Correct_Answer'].apply(get_openai_embedding)
        
        merged_df['Keyword_Recall_Score'] = merged_df.apply(
            lambda row: calculate_keyword_recall(row['Predicted_Answer'], row['Keywords']),
            axis=1
        )
        
        merged_df['Semantic_Similarity_Score'] = merged_df.apply(
            lambda row: calculate_semantic_similarity(row['Predicted_Embedding'], row['Correct_Embedding']),
            axis=1
        )
        
        merged_df['Mixed_Score'] = (
            0.6 * merged_df['Semantic_Similarity_Score']
            + 0.4 * merged_df['Keyword_Recall_Score']
        )
        
        return merged_df['Mixed_Score'].mean()
        
    except FileNotFoundError as e:
        print(f"오류: {e}. 파일 경로를 확인해주세요.")
        return 0.0