cat > setup_ram_models.sh <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

# ====== 사용자 조정 포인트 ======
# HF 캐시(디스크)는 /root 쪽(overlay)에 둬서 /workspace(20GB) 압박을 피함
export MODEL_CACHE="${MODEL_CACHE:-/root/.cache/huggingface}"

# 사용할 모델 repo id
LLM_REPO="LGAI-EXAONE/EXAONE-Deep-7.8B"
EMB_REPO="BAAI/bge-m3"
RER_REPO="BAAI/bge-reranker-v2-m3"

# RAM 디스크(휘발성) 위치
RAM_BASE="/dev/shm"
RAM_MODELS="$RAM_BASE/models"
RAM_TMP="$RAM_BASE/tmp"

# ====== 준비 ======
mkdir -p "$MODEL_CACHE" "$RAM_MODELS" "$RAM_TMP"

echo "== 캐시 위치: $MODEL_CACHE"
echo "== RAM 모델 위치: $RAM_MODELS"
echo "== RAM TMP: $RAM_TMP"

# ====== 헬퍼: 캐시에서 스냅샷 경로 찾기 ======
find_snapshot() {
  local repo="$1"
  local org="${repo%%/*}"
  local name="${repo##*/}"
  local base="$MODEL_CACHE/hub/models--${org//\//-}--${name//\//-}"
  if [[ ! -d "$base" ]]; then
    return 1
  fi
  # 최신 스냅샷 하나 선택
  local snap
  snap=$(ls -1dt "$base"/snapshots/* 2>/dev/null | head -n1 || true)
  [[ -z "$snap" ]] && return 1
  printf "%s" "$snap"
}

# ====== 헬퍼: rsync 있으면 rsync, 없으면 cp -a ======
copy_tree() {
  local src="$1" dst="$2"
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$src"/ "$dst"/
  else
    # cp는 삭제 동기화는 못하지만 일단 복사
    cp -a "$src"/. "$dst"/
  fi
}

# ====== 임베딩/리랭커를 RAM으로 복사 ======
echo "== 캐시에서 스냅샷 경로 탐색 중..."
emb_snap="$(find_snapshot "$EMB_REPO" || true)"
rer_snap="$(find_snapshot "$RER_REPO" || true)"

if [[ -z "$emb_snap" || -z "$rer_snap" ]]; then
  echo "!! 캐시에 스냅샷이 없습니다."
  echo "   먼저 온라인에서 다음처럼 캐시에만 받아 두세요:"
  echo "   HUGGINGFACE_HUB_CACHE=\$MODEL_CACHE huggingface-cli download $EMB_REPO --cache-dir \$MODEL_CACHE --local-dir-use-symlinks False --exclude 'tf_model.h5' 'flax_model.msgpack' 'pytorch_model.bin'"
  echo "   HUGGINGFACE_HUB_CACHE=\$MODEL_CACHE huggingface-cli download $RER_REPO --cache-dir \$MODEL_CACHE --local-dir-use-symlinks False --exclude 'tf_model.h5' 'flax_model.msgpack' 'pytorch_model.bin'"
  return 1 2>/dev/null || exit 1
fi

echo "== 복사: $EMB_REPO -> $RAM_MODELS/bge-m3"
copy_tree "$emb_snap" "$RAM_MODELS/bge-m3"

echo "== 복사: $RER_REPO -> $RAM_MODELS/bge-reranker-v2-m3"
copy_tree "$rer_snap" "$RAM_MODELS/bge-reranker-v2-m3"

# ====== 환경 변수 세팅 ======
# (우리 코드가 env 우선으로 읽도록 설계되어 있음)
export HF_HUB_OFFLINE=1
export HUGGINGFACE_HUB_CACHE="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE"
export HF_HOME="$MODEL_CACHE"
export TMPDIR="$RAM_TMP"

# 임베딩/리랭커를 로컬 디렉터리로 강제
export EMBEDDING_MODEL_NAME="$RAM_MODELS/bge-m3"
export RERANKER_MODEL_NAME="$RAM_MODELS/bge-reranker-v2-m3"

# (LLM은 캐시에서 로드: MODEL_NAME=repo-id 그대로, MODEL_CACHE가 /root를 가리킴)
# 필요 시 LLM을 로컬 디렉토리로 강제하고 싶다면:
# export LOCAL_DIR_EXAONE="/root/models/EXAONE-Deep-7.8B"  # 디스크 여유 있을 때만

# 요약 출력
echo
echo "== 설정 요약 =="
echo "HF offline         : $HF_HUB_OFFLINE"
echo "HF cache (disk)    : $MODEL_CACHE"
echo "TMPDIR (RAM)       : $TMPDIR"
echo "Embedding model    : $EMBEDDING_MODEL_NAME"
echo "Reranker model     : $RERANKER_MODEL_NAME"
echo
echo "이제 같은 셸에서:  python code/main.py"
EOF
chmod +x setup_ram_models.sh