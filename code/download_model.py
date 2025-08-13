import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. 공간이 넉넉한 경로를 명확하게 지정합니다 ---
CACHE_DIR = "/workspace/huggingface_cache"
MODEL_NAME = "upstage/SOLAR-10.7B-Instruct-v1.0"

# --- 2. 모든 라이브러리가 이 경로를 사용하도록 환경 변수를 설정합니다 ---
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR


print("="*50)
print("진단 스크립트를 시작합니다.")
print(f"사용할 캐시 디렉토리: {CACHE_DIR}")
print("="*50)


# --- 3. 캐시 디렉토리에 대한 쓰기 권한을 확인합니다 ---
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"✅ 디렉토리 '{CACHE_DIR}'가 존재하거나, 새로 생성되었습니다.")
    
    # 테스트 파일을 써서 권한이 있는지 최종 확인
    test_file_path = os.path.join(CACHE_DIR, "permission_test.txt")
    with open(test_file_path, "w") as f:
        f.write("This is a write test.")
    os.remove(test_file_path)
    print("✅ 캐시 디렉토리에 파일을 쓸 수 있는 권한이 확인되었습니다.")

except Exception as e:
    print(f"❌ 치명적 오류: 캐시 디렉토리를 생성하거나 파일에 쓸 수 없습니다.")
    print(f"오류 내용: {e}")
    print("이 문제가 해결되기 전까지는 모델을 다운로드할 수 없습니다.")
    exit() # 권한이 없으면 즉시 종료


# --- 4. 모델 다운로드를 시도합니다 ---
print("\n모델 다운로드를 시작합니다. 이 과정은 몇 분 정도 걸릴 수 있습니다...")
try:
    # 다른 기능 없이 오직 모델 다운로드만 실행
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
    )

    print("\n" + "="*50)
    print("✅✅✅ 성공! 모델이 성공적으로 다운로드 및 로드되었습니다. ✅✅✅")
    print("="*50)
    print("\n이제 원래의 rag.py 스크립트를 다시 실행해 보세요. 정상적으로 동작할 것입니다.")
    print("모델 파일은 이제 영구적으로 저장되었으므로 다시 다운로드하지 않습니다.")


except Exception as e:
    print("\n" + "!"*50)
    print("❌❌❌ 오류: 모델 다운로드에 실패했습니다. ❌❌❌")
    print("!"*50)
    print("\n오류의 원인은 다음과 같습니다:")
    print(e)
    print("\n이것은 코드의 문제가 아닌, 사용 중인 시스템 환경(RunPod)의 문제입니다.")
    print("RunPod 지원팀에 문의하여 이 오류 메시지를 보여주는 것이 좋을 수 있습니다.")

