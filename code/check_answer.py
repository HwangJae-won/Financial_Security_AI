import pandas as pd

# CSV 불러오기
sample_df = pd.read_csv("data/sample.csv")  # ID, 정답
submission_df = pd.read_csv("results/solar_more_large_model_rag.csv")  # ID, Answer

# ID 기준으로 병합
merged = pd.merge(sample_df, submission_df, on="ID", how="inner")

# 비교 컬럼 생성
merged["정답여부"] = merged.apply(
    lambda row: str(row["정답"]).strip() == str(row["Answer"]).strip(),
    axis=1
)

# 결과 출력
print(merged)

# 정답률 계산
print("정답개수:", merged["정답여부"].sum())
accuracy = merged["정답여부"].mean() * 100
print(f"정답률: {accuracy:.2f}%")