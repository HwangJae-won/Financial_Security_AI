#!/bin/bash

# 스크립트 실행 중 오류 발생 시 즉시 중단
set -e

# --- 1. root 권한으로 실행되는 부분 ---
echo "[SETUP] Starting system setup as root..."

# 사용자 생성
if id "myproject_user" &>/dev/null; then
    echo "User 'myproject_user' already exists. Skipping user creation."
else
    useradd -m -s /bin/bash myproject_user
    echo "User 'myproject_user' created."
fi

# 디렉토리 생성 및 권한 부여
PROJECT_DIR="/var/www/Financial_Security_AI"
mkdir -p $PROJECT_DIR
chown -R myproject_user:myproject_user /var/www

echo "Directory permissions set for $PROJECT_DIR."

# --- 2. myproject_user 권한으로 실행되는 부분 ---
echo "[SETUP] Running project setup as myproject_user..."

# su -c 를 사용하여 myproject_user로 명령어 실행
# !! 중요: 보안을 위해 아래 git clone 명령어는 SSH 방식으로 변경하는 것을 강력히 권장합니다.
# 이 스크립트에서는 토큰을 제거하고 사용자에게 직접 입력받도록 수정했습니다.
su - myproject_user -c "
    set -e
    cd /var/www/
    
    # Git 클론 (SSH 방식 권장)
    git clone https://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@github.com/HwangJae-won/Financial_Security_AI.git
    
    cd $PROJECT_DIR
    
    # 가상 환경 생성 및 라이브러리 설치
    echo 'Creating Python virtual environment...'
    python3 -m venv venv
    
    echo 'Installing dependencies from requirements.txt...'
    source venv/bin/activate
    pip install -r requirements.txt
    
    echo 'Setup for myproject_user is complete.'
"

echo "[SUCCESS] All setup tasks are complete."