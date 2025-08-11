import pandas as pd
from tqdm import tqdm
from model import load_model
from prompt import make_prompt_auto
from utils import extract_answer_only
import os


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
        prompt = make_prompt_auto(q, version=PROMPT_VERSION)
        stage_result = None  # 성공 단계 기록
    
        # 1) 1차: 보수적 (greedy)
        out = pipe(prompt, max_new_tokens=256, temperature=0.0)  # do_sample=False(default)
        raw = out[0]["generated_text"]
        ans = extract_answer_only(raw, original_question=q, prompt=prompt)
        if ans not in ("0", "미응답"):
            stage_result = "1차 성공"
        else:
            print(f"\n[문항 {idx}] 1차 실패 → 샘플링(온건) 재시도")
    
        # 2) 실패면: 샘플링으로 여러 개 뽑아 유효한 것 고르기
        if stage_result is None:
            outs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.95,
                num_return_sequences=3, repetition_penalty=1.05)
            picked = None
            for o in outs:
                cand = extract_answer_only(o["generated_text"], original_question=q, prompt=prompt)
                if cand not in ("0", "미응답"):
                    picked = cand
                    raw = o["generated_text"]
                    break
            if picked:
                ans = picked
                stage_result = "2차 성공"
            else:
                print(f"[문항 {idx}] 2차 실패 → 샘플링(넓게) 재시도")
    
        # 3) 그래도 실패면: 더 넓게 샘플링
        if stage_result is None:
            outs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.8, top_p=1.0,
                num_return_sequences=5, repetition_penalty=1.05)
            picked = None
            for o in outs:
                cand = extract_answer_only(o["generated_text"], original_question=q, prompt=prompt)
                if cand not in ("0", "미응답"):
                    picked = cand
                    raw = o["generated_text"]
                    break
            if picked:
                ans = picked
                stage_result = "3차 성공"
            else:
                ans = "0"  # 마지막 가드
                stage_result = "실패"
    
        # 최종 결과 출력
        print(f"\n[문항 {idx}] 단계 결과: {stage_result}")
        print(f"[문항 {idx}] 최종 추출된 답변: {ans}")
        preds.append(ans)
        

    print("\n✅ 추론 완료!")
    print(f"생성된 답변 개수: {len(preds)}")
    
    experiment_name = "solar_postcessing.csv"
    print("📄 제출 파일 생성 중...")
    sample_submission = pd.read_csv(DATA_PATH + "sample_submission.csv")
    sample_submission['Answer'] = preds
    os.makedirs(OUTPUT_PATH, exist_ok=True)
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
    
