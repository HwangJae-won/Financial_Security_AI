import os
from eval_multiple import calculate_objective_accuracy_details
from eval_subjective import calculate_subjective_score

if __name__ == '__main__':
    SUBMISSION_FILE = "results/fit_condition_solar.csv"
    
    print("--- 객관식 점수 계산 ---")
    objective_results = calculate_objective_accuracy_details(SUBMISSION_FILE)
    objective_score = objective_results["combined_results"]["combined_accuracy"] / 100
    print(f"✅ 객관식 점수: {objective_score:.4f}\n")

    print("--- 주관식 점수 계산 ---")
    subjective_score = calculate_subjective_score(SUBMISSION_FILE)
    print(f"✅ 주관식 점수: {subjective_score:.4f}\n")
    
    # 최종 스코어 계산 (0.5 vs 0.5)
    final_score = (objective_score * 0.5) + (subjective_score * 0.5)
    
    print("--- 최종 예상 스코어 ---")
    print(f"객관식 점수(50%) : {objective_score * 0.5:.4f}")
    print(f"주관식 점수(50%) : {subjective_score * 0.5:.4f}")
    print(f"최종 스코어     : {final_score:.4f}")