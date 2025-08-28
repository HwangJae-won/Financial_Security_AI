# (A) new-impl에서 코드 파일 위치 찾기
# 너가 옮겼다는 새 위치에 맞춰 경로만 확인해서 사용해.
# 예) /workspace/models/models--Alibaba-NLP--new-impl/snapshots/<hash>/
NEW_IMPL_SNAP=$(ls -d /workspace/models/models--Alibaba-NLP--new-impl/snapshots/* | head -n1)

# (B) 두 파일을 리랭커 모델 디렉터리로 복사
cp "$NEW_IMPL_SNAP/configuration.py" /workspace/models/gte-multilingual-reranker-base/
cp "$NEW_IMPL_SNAP/modeling.py"       /workspace/models/gte-multilingual-reranker-base/

# (C) config.json의 auto_map을 로컬 파일로 수정
python - <<'PY'
import json, os
p = "/workspace/models/gte-multilingual-reranker-base/config.json"
with open(p, "r", encoding="utf-8") as f:
    cfg = json.load(f)

am = cfg.get("auto_map", {})
def fix(v):
    if isinstance(v, str):
        v = v.replace("Alibaba-NLP/new-impl--configuration", "configuration")
        v = v.replace("Alibaba-NLP/new-impl--modeling", "modeling")
    return v

if isinstance(am, dict):
    for k in list(am.keys()):
        am[k] = fix(am[k])
    cfg["auto_map"] = am

# 불필요한 원격 커밋 힌트 제거(오프라인 안전)
for k in ["code_revision","trust_remote_code_revision","_commit_hash"]:
    cfg.pop(k, None)

with open(p, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
print("[OK] patched", p)
PY

# (D) 오프라인 스모크 테스트
export HF_HUB_OFFLINE=1
python - <<'PY'
from sentence_transformers import CrossEncoder
m = CrossEncoder("/workspace/models/gte-multilingual-reranker-base",
                 trust_remote_code=True, device="cpu", max_length=16)
print("Reranker OK (local patched)")
PY