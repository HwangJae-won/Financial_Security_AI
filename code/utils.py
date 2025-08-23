import re
from typing import List, Tuple, Dict, Optional
OPTION_PATTERN = re.compile(r"^\s*[1-9][0-9]?\s")

def is_multiple_choice(question_text):
    """
    객관식 여부와 선택지 개수 반환
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(OPTION_PATTERN.match(line)) for line in lines)
    return option_count >= 2, option_count

def extract_question_and_choices(full_text):
    """
    질문과 선택지 분리
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []
    for line in lines:
        if OPTION_PATTERN.match(line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    question = " ".join(q_lines)
    return question, options

def extract_answer_only(generated_text: str, original_question: str, prompt: str) -> str:
    """
    - "답변:" 이후 텍스트만 추출
    - 객관식: 정답 숫자만 추출 (실패 시 '0' 반환)
    - 주관식: 전체 텍스트 반환
    - 빈 응답은 '미응답' 반환
    """
    # 프롬프트 제거
    text = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()
    # "답변:" 이후만 추출
    if "답변:" in text:
        text = text.split("답변:")[-1].strip()
    if not text:
        return "미응답"
    if '---' in text:
        text = text.split('---')[0].strip()
    
    if '### 생각의 과정' in text:
        text = text.split('### 생각의 과정')[0].strip()

    is_mc, option_count = is_multiple_choice(original_question)
    if is_mc:
        match = re.match(r"\D*([1-9][0-9]?)", text)
        if match:
            num = int(match.group(1))
            return str(num) if 1 <= num <= option_count else '0'
        return '0'
    return text


def clean_markdown(text: str) -> str:
    """
    텍스트에서 볼드체, 이탤릭체 등 마크다운 특수문자를 제거합니다.
    """
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) # **text** 제거
    text = re.sub(r'_(.*?)_', r'\1', text)     # _text_ 제거
    text = re.sub(r'~(.*?)~', r'\1', text)     # ~text~ 제거
    return text



def load_and_chunk_file(file_path: str, filename: str):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".pdf":
        raw_text = load_pdf_text(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    if not raw_text.strip():
        print(f"⚠️ 경고: '{file_path}'에서 텍스트를 추출하지 못했습니다.")
        return []
    chunks, labels = chunk_law_text(_clean_text(raw_text))
    file_metadata = _extract_metadata_from_filename(filename)
    return [
        Document(page_content=chunk, metadata={**file_metadata, "source": file_path, "label": label})
        for chunk, label in zip(chunks, labels)
    ]
def load_all_documents(folder_path: str):
    all_documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension in [".pdf", ".txt"]:
            print(f"📄 로딩 중: {file_path}")
            docs = load_and_chunk_file(file_path, filename)
            all_documents.extend(docs)
        else:
            print(f"⚠️ 경고: 지원되지 않는 파일 형식 스킵 - {file_path}")
    return all_documents


def extract_metadata_from_filename(filename: str) -> Dict[str, Optional[str]]:
    """
    다양한 파일명 패턴에서 법률 메타데이터를 추출합니다.
    (이 함수는 setup_retriever에서만 사용됩니다.)
    """
    # 1. 가장 흔한 패턴: (유형)(제...호)(날짜)
    pattern1 = re.compile(r'(.+?)\((.+?)\)\(제(.+?호)\)\((\d+)\)')
    match1 = pattern1.search(filename)
    if match1:
        return {
            "law_name": match1.group(1).strip(),
            "law_type": match1.group(2),
            "law_number": match1.group(3),
            "enactment_date": match1.group(4)
        }

    # 2. 번호가 없는 경우를 대비한 패턴: (유형).pdf
    pattern2 = re.compile(r'(.+?)\((.+?)\)\.pdf')
    match2 = pattern2.search(filename)
    if match2:
        return {
            "law_name": match2.group(1).strip(),
            "law_type": match2.group(2),
            "law_number": None,
            "enactment_date": None
        }

    # 3. 그 외 알 수 없는 형식을 대비한 최후의 패턴
    pattern3 = re.compile(r'(.+?)\.pdf')
    match3 = pattern3.search(filename)
    if match3:
        return {
            "law_name": match3.group(1).strip(),
            "law_type": None,
            "law_number": None,
            "enactment_date": None
        }
    
    print(f"⚠️ 경고: 파일명 패턴 불일치 - {filename}")
    return {}