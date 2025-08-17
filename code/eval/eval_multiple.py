import pandas as pd
import os
from typing import Dict, Any

def calculate_objective_accuracy_details(submission_file: str) -> Dict[str, Any]:
    """
    두 개의 정답 데이터셋을 사용하여 제출 파일의 개별 및 종합 정답률을 계산하는 함수.

    Args:
        submission_file (str): 모델의 답변이 포함된 제출 파일의 경로.

    Returns:
        Dict[str, Any]: 각 데이터셋의 분석 결과와 종합 결과를 담은 딕셔너리.
    """
    
    sample_files = {
        "sample1": "data/sample1.csv", 
        "sample2": "data/sample2.csv"
    }
    
    results = {
        "individual_results": {},
        "combined_results": {
            "total_correct": 0,
            "total_questions": 0,
            "combined_accuracy": 0.0
        }
    }

    try:
        submission_df = pd.read_csv(submission_file)
    except FileNotFoundError:
        print(f"오류: 제출 파일 '{submission_file}'을 찾을 수 없습니다.")
        return results

    for name, sample_file in sample_files.items():
        try:
            sample_df = pd.read_csv(sample_file)
            
            merged = pd.merge(sample_df, submission_df, on="ID", how="inner")
            
            merged["정답여부"] = merged.apply(
                lambda row: str(row["정답"]).strip() == str(row["Answer"]).strip(),
                axis=1
            )
            
            correct_count = merged["정답여부"].sum()
            total_questions = len(merged)
            accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0.0
            
            results["individual_results"][name] = {
                "correct_answers": correct_count,
                "total_questions": total_questions,
                "accuracy": accuracy
            }
            
            results["combined_results"]["total_correct"] += correct_count
            results["combined_results"]["total_questions"] += total_questions
            
        except FileNotFoundError:
            print(f"경고: 정답 파일 '{sample_file}'을 찾을 수 없어 스킵합니다.")
            continue
            
    if results["combined_results"]["total_questions"] > 0:
        results["combined_results"]["combined_accuracy"] = (
            results["combined_results"]["total_correct"] / 
            results["combined_results"]["total_questions"]
        ) * 100

    return results