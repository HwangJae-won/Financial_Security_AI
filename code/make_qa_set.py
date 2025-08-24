import os
import pdfplumber
import pandas as pd
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
# 사용자 제공 함수s
def load_pdf_text(pdf_path: str) -> str:
    """
    PDF 파일에서 모든 페이지의 텍스트를 추출
    """
    texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                texts.append(t)
        print(f"'{os.path.basename(pdf_path)}' 파일에서 텍스트 추출 완료.")
        return "\n".join(texts)
    except Exception as e:
        print(f"'{os.path.basename(pdf_path)}' 파일 처리 중 오류 발생: {e}")
        return ""

def generate_test_set_with_hf_model(text: str, model_name: str, target_count: int) -> List[Dict[str, str]]:
    """
    허깅페이스 모델을 사용하여 객관식, 주관식 혼합 테스트 셋을 생성합니다.
    """
    qa_pairs = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"모델 실행 장치: {device}")
    
    if device == "cpu":
        print("경고: GPU가 없어 CPU로 실행됩니다. 매우 느릴 수 있습니다.")
    
    try:
        print(f"'{model_name}' 모델 로딩 중...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            cache_dir='/dev/shm/huggingface_cache', # 캐시 경로 직접 지정
            torch_dtype=torch.float16,
            device_map='auto', 
            trust_remote_code=True
        )
        print("모델 로딩 완료.")
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        return []

    instruction = (
        f"다음 금융 보안 관련 문서를 기반으로 객관식 및 주관식 문제를 {target_count}개 생성해줘. "
        "문제와 정답은 '문제: ... 정답: ...' 형식으로 명확하게 구분해줘."
        "객관식 문제의 경우, 정답은 '정답: (번호)' 형식으로 표시하고, 주관식 문제의 경우, '정답: (상세 답변)' 형식으로 표시해줘."
    )
    
    max_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 2048
    chunk_size = max_length - len(tokenizer.encode(instruction)) - 512
    
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"\n원본 텍스트가 {len(chunks)}개의 청크로 분할되었습니다.")

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            print(f"경고: 텍스트 청크 {i+1}이 비어있습니다. 건너뜁니다.")
            continue
        print(f"\n--- 텍스트 청크 {i+1}/{len(chunks)} 처리 시작 ---")
        
        chunk_prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{chunk}\n\n### Response:"
        
        inputs = tokenizer(chunk_prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=len(input_ids[0]) + 512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"\n모델이 생성한 원본 텍스트:\n{generated_text}\n")
            
            try:
                generated_qa_str = generated_text.split("### Response:")[1].strip()
                lines = generated_qa_str.split('\n')
                
                current_question = ""
                current_answer = ""
                for line in lines:
                    if line.startswith("문제:"):
                        if current_question and current_answer:
                            qa_pairs.append({"Question": current_question.strip(), "Answer": current_answer.strip()})
                        current_question = line.replace("문제:", "", 1).strip()
                        current_answer = ""
                    elif line.startswith("정답:"):
                        current_answer += line.replace("정답:", "", 1).strip()
                    elif current_question:
                        current_answer += " " + line.strip()
                
                if current_question and current_answer:
                    qa_pairs.append({"Question": current_question.strip(), "Answer": current_answer.strip()})
            except IndexError:
                print("경고: 모델 출력 형식이 '### Response:'를 포함하지 않아 파싱을 건너뜁니다.")
                continue
        except Exception as e:
            print(f"모델 생성 중 오류 발생: {e}")
            continue

    print(f"\n\n--- 최종 결과 ---")
    print(f"총 {len(qa_pairs)}개의 질문-답변 쌍 생성 완료.")
    return qa_pairs

def save_qa_to_csv(qa_data: List[Dict[str, str]], output_filename: str):
    """생성된 질문-답변 쌍을 CSV 파일로 저장합니다."""
    if not qa_data:
        print("저장할 데이터가 없습니다.")
        return
    
    df = pd.DataFrame(qa_data)
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n성공적으로 '{output_filename}' 파일에 질문-답변 쌍을 저장했습니다.")

if __name__ == "__main__":
    pdf_folder = "./laws"
    model_id = "yanolja/EEVE-Korean-Instruct-2.8B-v1.0"
    target_qa_count = 200
    
    full_corpus = ""
    if os.path.isdir(pdf_folder):
        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                file_path = os.path.join(pdf_folder, filename)
                full_corpus += load_pdf_text(file_path) + "\n\n"
    else:
        print(f"오류: 지정된 폴더 '{pdf_folder}'가 존재하지 않습니다.")

    if full_corpus.strip():
        print("테스트 셋 생성을 시작합니다...")
        test_set = generate_test_set_with_hf_model(full_corpus, model_id, target_qa_count)
        save_qa_to_csv(test_set, "generated_test_set.csv")
    else:
        print("텍스트를 추출할 PDF 파일이 없어 테스트 셋을 생성할 수 없습니다.")