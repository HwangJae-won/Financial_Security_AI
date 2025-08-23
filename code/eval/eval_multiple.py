import pandas as pd
import os
from typing import Dict, Any

def calculate_objective_accuracy_details(submission_file: str) -> Dict[str, Any]:
    """
    단일 정답 데이터셋(answer.csv)을 사용하여 제출 파일의 정답률을 계산하는 함수.

    Args:
        submission_file (str): 모델의 답변이 포함된 제출 파일의 경로.

    Returns:
        Dict[str, Any]: 분석 결과를 담은 딕셔너리.
    """
    answer_file = "data/answer.csv"
    results = {
        "correct_answers": 0,
        "total_questions": 0,
        "accuracy": 0.0
    }

    try:
        submission_df = pd.read_csv(submission_file)
        answer_df = pd.read_csv(answer_file)
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. {e}")
        return results

    merged = pd.merge(answer_df, submission_df, on="ID", how="inner")

    merged["정답여부"] = merged.apply(
        lambda row: str(row["정답"]).strip() == str(row["Answer"]).strip(),
        axis=1
    )

    correct_count = merged["정답여부"].sum()
    total_questions = len(merged)
    accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0.0

    results["correct_answers"] = correct_count
    results["total_questions"] = total_questions
    results["accuracy"] = accuracy

    return results