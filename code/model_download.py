# code/model_download.py
import os
import sys
from pathlib import Path
from typing import Optional
from huggingface_hub import snapshot_download

MODELS_DIR = "/workspace/models"

MODELS = [
    {
        "name": "EXAONE LLM",
        "repo": "LGAI-EXAONE/EXAONE-Deep-7.8B",
        "dest": f"{MODELS_DIR}/EXAONE-Deep-7.8B",
        "revision": "17b70148e344c28f54a542a030a805b8c96be8c3",
    },
    {
        "name": "E5 embeddings",
        "repo": "intfloat/multilingual-e5-small",
        "dest": f"{MODELS_DIR}/multilingual-e5-small",
        "revision": "c007d7ef6fd86656326059b28395a7a03a7c5846",
    },
    {
        "name": "GTE Reranker (multilingual)",
        "repo": "Alibaba-NLP/gte-multilingual-reranker-base",
        "dest": f"{MODELS_DIR}/gte-multilingual-reranker-base",
        "revision": "8215cf04918ba6f7b6a62bb44238ce2953d8831c",
    },
    {
        # GTE Reranker가 의존하는 동적 모듈 코드
        "name": "GTE Reranker dynamic code",
        "repo": "Alibaba-NLP/new-impl",
        "dest": f"{MODELS_DIR}/new-impl",
        "revision": None
    },
]

def has_weights(dirpath: Path) -> bool:
    # safetensors or pytorch_model*.bin anywhere under dir
    if not dirpath.exists():
        return False
    if list(dirpath.rglob("*.safetensors")):
        return True
    if list(dirpath.rglob("pytorch_model*.bin")):
        return True
    return False

def has_config(dirpath: Path) -> bool:
    # transformers/sentence-transformers 공통으로 config.json이 거의 항상 존재
    return dirpath.joinpath("config.json").exists() or bool(list(dirpath.rglob("config.json")))

def is_present(dest: str) -> bool:
    p = Path(dest)
    # reranker 코드는 가중치가 없을 수도 있으므로 has_weights 조건 제거
    if "new-impl" in str(p):
        return p.exists() and has_config(p) and bool(list(p.rglob("*.py")))
    else:
        return p.exists() and has_config(p) and has_weights(p)

def ensure_online_once():
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        sys.stderr.write(
            "ERROR: HF_HUB_OFFLINE=1 상태에서는 최초/추가 다운로드가 불가합니다.\n"
            "       (이미 폴더가 완비되어 있으면 이 스크립트는 아무 것도 하지 않고 통과합니다.)\n"
        )
        sys.exit(1)

def download(repo_id: str, dest: str, revision: Optional[str] = None):
    print(f"==> Downloading {repo_id} -> {dest}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=dest,
        local_dir_use_symlinks=False,
        resume_download=True,
        # 필요한 것만!
        allow_patterns=[
            "config.json",
            "pytorch_model.bin",        # 또는 model.safetensors
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "*.py",                     # 동적 모듈 코드
            "README.md",
            ".gitattributes",
        ],
        # 무거운 것 전부 제외
        ignore_patterns=[
            "*.onnx", "*int8*.onnx", "*uint8*.onnx", "*fp16*.onnx", "*q4*.onnx", "*bnb*.onnx",
            "*.tflite", "*.gguf", "*.safetensors.index.json"
        ],
    )
    print(f"✔ Saved to {dest}")

def main():
    force = "--force" in sys.argv
    needs_download = []

    # 1) 어떤 게 비어있는지 먼저 점검
    for m in MODELS:
        present = is_present(m["dest"])
        if present and not force:
            print(f"[SKIP] {m['name']} already present at {m['dest']}")
        else:
            needs_download.append(m)

    # 2) 다운로드 필요 없으면 종료(오프라인이어도 OK)
    if not needs_download:
        print("All required models are present. Nothing to do.")
        return

    # 3) 다운로드 필요한데 오프라인이면 에러
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        missing = ", ".join(m["name"] for m in needs_download)
        sys.stderr.write(
            f"ERROR: Offline 모드(HF_HUB_OFFLINE=1)인데 다음 모델이 비어있습니다: {missing}\n"
            "       온라인으로 전환한 뒤 다시 실행하거나, 다른 머신에서 폴더를 복사하세요.\n"
        )
        sys.exit(1)

    # 4) 필요한 것만 다운로드
    for m in needs_download:
        Path(m["dest"]).mkdir(parents=True, exist_ok=True)
        download(m["repo"], m["dest"], m["revision"])

    print("\nDone. 이제 오프라인(HF_HUB_OFFLINE=1)로 추론이 가능합니다.")

if __name__ == "__main__":
    main()