import pandas as pd
from tqdm import tqdm
from model import load_model
from prompt import make_prompt_auto
from utils import extract_answer_only

DATA_PATH = "data/"
OUTPUT_PATH = "results/"
PROMPT_VERSION = "v2" 

def main():
    print("📂 테스트 데이터 로딩 중...")
    test = pd.read_csv(DATA_PATH + 'test.csv')
    print(f"✅ 테스트 데이터 로드 완료! 총 문항 수: {len(test)}")

    print("🧠 모델 로딩 중...")
    pipe = load_model()
    print("✅ 모델 로딩 완료!")

    preds = []
    print("🚀 추론 시작!")
    for idx, q in enumerate(tqdm(test['Question'], desc="Inference")):
        prompt = make_prompt_auto(q, version = PROMPT_VERSION)
        print(f"\n[문항 {idx+1}] 프롬프트 생성 완료:\n{prompt[:100]}...")  # 프롬프트 일부 출력
        output = pipe(prompt, max_new_tokens=256, temperature=0.2, top_p=0.9)
        print(f"[문항 {idx+1}] 모델 출력:\n{output[0]['generated_text']}")  # 출력 일부
        pred_answer = extract_answer_only(output[0]["generated_text"], original_question=q, prompt=prompt)
        print(f"[문항 {idx+1}] 추출된 답변: {pred_answer}")
        preds.append(pred_answer)

    print("\n✅ 추론 완료!")
    print(f"생성된 답변 개수: {len(preds)}")
    
    experiment_name = "solar_postcessing.csv"
    print("📄 제출 파일 생성 중...")
    sample_submission = pd.read_csv(DATA_PATH + "sample_submission.csv")
    sample_submission['Answer'] = preds
    sample_submission.to_csv(OUTPUT_PATH + experiment_name, index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {OUTPUT_PATH + experiment_name}")

if __name__ == "__main__":
    print("=== 금융보안 AI 추론 파이프라인 시작 ===")
    main()
    print("=== 모든 작업 완료 ===")


    # BATCH_SIZE = 8
    # preds = []
    # prompts = [PROMPT_FUNC(q) for q in tqdm(test['Question'], desc="프롬프트 생성")]
    # batches = [prompts[i:i + BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]

    # with tqdm(batches, desc="추론 진행") as pbar:
    #     for batch_prompts in pbar:
    #         outputs = pipe(
    #             batch_prompts,
    #             max_new_tokens=128,
    #             temperature=0.2,
    #             top_p=0.9,
    #             batch_size=BATCH_SIZE
    #         )
    #         for i, batch_output in enumerate(outputs):
    #             output_text = batch_output[0]["generated_text"]
    #             current_question_index = (pbar.n - 1) * BATCH_SIZE + i
    #             original_question = test['Question'].iloc[current_question_index]
    #             pred_answer = extract_answer_only(output_text, original_question=original_question)
    #             preds.append(pred_answer)

    # print("\n✅ 추론 완료!")
    # print(f"생성된 답변 개수: {len(preds)}")
    