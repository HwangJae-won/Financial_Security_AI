import pandas as pd
from tqdm import tqdm
from model import load_model
from prompt import make_prompt_fewshot
from utils import extract_answer_only

DATA_PATH = "data/"
OUTPUT_PATH = "results/"
PROMPT_FUNC = make_prompt_fewshot 

def main():
    test = pd.read_csv(DATA_PATH + 'test.csv')
    pipe = load_model()
    preds = []

    for q in tqdm(test['Question'], desc="Inference"):
        prompt = PROMPT_FUNC(q)
        output = pipe(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9)
        pred_answer = extract_answer_only(output[0]["generated_text"], original_question=q)
        preds.append(pred_answer)
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
    
    experiment_name = "baseline_batch.csv"
    sample_submission = pd.read_csv(DATA_PATH + "sample_submission.csv")
    sample_submission['Answer'] = preds
    sample_submission.to_csv(OUTPUT_PATHOUTPUT_PATH+ experiment_name, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()