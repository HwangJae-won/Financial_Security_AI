#!/usr/bin/env bash
set -Eeuo pipefail

# """
# 주관식 답변 list
# python code/main.py ask --question "전자자금이체의 지급 효력 발생 시점을 전자금융거래법 기준에 따라 설명하세요."
# python code/main.py ask --question "금융회사가 정보보호 예산을 관리할 때, 전자금융감독규정상 정보기술부문 인력 및 예산의 기준 비율은 얼마인가요?"
# python code/main.py ask --question "전자금융거래법에 따라 이용자가 금융 분쟁조정을 신청할 수 있는 기관을 기술하세요."
# python code/main.py ask --question "DNSSEC의 주요 기능과 이를 통해 해결할 수 있는 보안 위협을 설명하세요."
# python code/main.py ask --question "딥페이크 기술 악용에 대비한 금융권의 대응 방안을 기술하세요."
# python code/main.py ask --question "디지털 지갑(Digital Wallet)에서 우려되는 주요 보안 위협을 설명하세요."
# python code/main.py ask --question "정보보호의 3대 요소에 해당하는 보안 목표를 3가지 기술하세요."

# """




# # 프로젝트 루트로 이동
# cd "$(dirname "$0")"

# 0) (선택) venv 활성화가 필요하면 주석 해제
# source venv/bin/activate

# 1) RAM 세팅: 임베딩/리랭커는 /dev/shm, LLM(EXAONE)은 /root 캐시 사용
if [[ -f "./setup_ram_models.sh" ]]; then
  # setup_ram_models.sh는 환경변수까지 잡아줌
  source ./setup_ram_models.sh
else
  echo "⚠️ setup_ram_models.sh 가 없습니다. 먼저 만들어 둔 스크립트를 생성/검토하세요."
  echo "   (그래도 계속 진행하지만, 임베딩/리랭커가 RAM으로 올라가지 않을 수 있습니다)"
  # 최소 필수 환경(LLM은 /root 캐시, 임시파일은 RAM)
  export MODEL_CACHE="${MODEL_CACHE:-/root/.cache/huggingface}"
  export HF_HUB_OFFLINE=1
  export HUGGINGFACE_HUB_CACHE="$MODEL_CACHE"
  export TRANSFORMERS_CACHE="$MODEL_CACHE"
  export HF_HOME="$MODEL_CACHE"
  export TMPDIR="/dev/shm/tmp"; mkdir -p "$TMPDIR"
fi

# 2) 하위 커맨드 파싱
usage() {
  cat <<USAGE
Usage:
  bash script.sh build_all                # laws/ 와 supplement/ 인덱스 빌드
  bash script.sh ask "질문 내용"           # 단일 질문 추론
  bash script.sh run data/test.csv        # CSV 일괄 추론 (컬럼명: Question)
USAGE
}

cmd="${1:-}"; shift || true

case "$cmd" in
  build_all)
    echo "📦 인덱스 빌드 시작..."
    # laws/ 는 법령 패턴
    if [[ -d "laws" ]]; then
      python code/main.py build --dir "laws/" --kind "law"
    else
      echo "⚠️ laws/ 폴더가 없습니다. 건너뜁니다."
    fi
    # supplement/ 는 일반 문서(있을 때만)
    if [[ -d "supplement" ]]; then
      python code/main.py build --dir "supplement/" --kind "generic"
    else
      echo "ℹ️ supplement/ 폴더가 없어 laws 인덱스만 사용합니다."
    fi
    echo "✅ 인덱스 빌드 완료"
    ;;

  ask)
    question="${1:-}"
    if [[ -z "$question" ]]; then
      echo "❌ 질문이 비었습니다."; usage; exit 1
    fi
    echo "❓ 질문: $question"
    python code/main.py ask --question "$question"
    ;;

  run)
    csv_path="${1:-}"
    if [[ -z "$csv_path" || ! -f "$csv_path" ]]; then
      echo "❌ CSV 경로가 없거나 파일이 존재하지 않습니다: $csv_path"; usage; exit 1
    fi
    echo "🧪 CSV 일괄 추론 시작: $csv_path"
    python code/main.py run --csv "$csv_path"
    echo "✅ 완료.sv"
    ;;

  *)
    usage; exit 1
    ;;
esac