import sys
import os
import tempfile
import pandas as pd
from tqdm import tqdm
import nltk

# --- START: NLTK 데이터 다운로드 ---
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    print("NLTK 데이터 'averaged_perceptron_tagger_eng'를 다운로드합니다...")
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    print("✅ NLTK 데이터 다운로드 완료.")
# --- END: NLTK 데이터 다운로드 ---

# --- START: 환경 문제 해결을 위한 최종 설정 ---
CACHE_DIR = "/workspace/huggingface_cache"
TMP_DIR = "/workspace/tmp"
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
os.environ['TMPDIR'] = TMP_DIR
os.environ['TEMP'] = TMP_DIR
os.environ['TMP'] = TMP_DIR
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
tempfile.tempdir = TMP_DIR
print(f"✅ 모든 캐시 및 임시 파일 경로가 '/workspace'로 설정되었습니다.")
# --- END: 환경 문제 해결을 위한 최종 설정 ---

# 필요한 라이브러리들을 import 합니다.
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader # PyPDFLoader 추가
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download

# utils.py가 같은 code 폴더 안에 있다고 가정합니다.
from utils import extract_answer_only, is_multiple_choice

# --- START: 모델 로딩 기능 (GGUF 안정화 버전) ---

MODEL_NAME = "TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF"
MODEL_FILE = "solar-10.7b-instruct-v1.0.Q4_K_M.gguf" # 4-bit 양자화 버전 (약 6.5GB)

def load_llm():
    """
    지정된 GGUF 모델을 다운로드하고 안정적인 설정으로 로드합니다.
    """
    print(f"'{MODEL_NAME}'에서 모델 파일을 다운로드합니다...")
    model_path = hf_hub_download(
        repo_id=MODEL_NAME,
        filename=MODEL_FILE,
        cache_dir=CACHE_DIR,
    )
    print(f"✅ 모델 파일 다운로드 완료: {model_path}")
    
    print("LlamaCpp를 사용하여 모델을 로딩합니다 (안정화 설정)...")
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=True,
    )
    print("✅ GGUF 모델 로딩 완료!")
    return llm

# --- END: 모델 로딩 기능 ---

# --- START: RAG 준비 기능 (PDF 지원 추가) ---

def setup_retriever(file_path="data/law.txt", embedding_model_name="jhgan/ko-sroberta-multitask"):
    """
    문서(txt 또는 pdf)를 로드하고 검색기(Retriever)를 설정하는 함수
    """
    print(f"📄 문서를 로딩하고 벡터화합니다: {file_path}")
    
    # --- START: 수정된 부분 ---
    # 파일 확장자에 따라 적절한 로더를 선택합니다.
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".txt":
        loader = UnstructuredFileLoader(file_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")
    # --- END: 수정된 부분 ---

    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    print(f"✨ 임베딩 모델을 로딩합니다: {embedding_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    print("🔍 벡터 저장소(FAISS)를 생성합니다...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("✅ 벡터 저장소 생성 완료.")
    
    return vectorstore.as_retriever()

# --- END: RAG 준비 기능 ---

# --- START: 메인 추론 로직 ---
def main():
    """RAG를 적용하여 test.csv에 대한 추론을 수행하는 메인 함수"""

    # 1. RAG 및 LLM 초기 설정 시작
    print("--- 1. RAG 및 LLM 초기 설정 시작 ---")
    
    # 💡 수정할 부분
    FAISS_INDEX_PATH = "faiss_index"
    embedding_model_name = "jhgan/ko-sroberta-multitask"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # 이미 저장된 벡터 저장소가 있는지 확인
    if os.path.exists(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH):
        print("✅ 기존 벡터 저장소(FAISS)를 불러옵니다...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("🔍 새로운 벡터 저장소(FAISS)를 생성합니다...")
        
        # 문서 로딩 및 분할 (기존 코드와 동일)
        laws_folder_path = "laws/"
        all_documents = []
        if os.path.exists(laws_folder_path):
            for filename in os.listdir(laws_folder_path):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(laws_folder_path, filename)
                    print(f"📄 로딩 중: {file_path}")
                    loader = PyPDFLoader(file_path)
                    all_documents.extend(loader.load())
        else:
            raise FileNotFoundError(f"'{laws_folder_path}' 폴더를 찾을 수 없습니다.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(all_documents)
        
        # 벡터 저장소 생성 및 저장
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"✅ 새로운 벡터 저장소가 '{FAISS_INDEX_PATH}' 폴더에 저장되었습니다.")
        
    retriever = vectorstore.as_retriever()
    llm = load_llm()
    print("--- ✅ RAG 및 LLM 초기 설정 완료 ---\n")

    # 2. 테스트 데이터 로드
    print("--- 2. 테스트 데이터 로딩 시작 ---")
    DATA_PATH = "data/"
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    print(f"✅ 테스트 데이터 로드 완료! 총 문항 수: {len(test_df)}")
    print("--- ✅ 테스트 데이터 로딩 완료 ---\n")

    # 3. 추론 실행
    print("--- 3. RAG 기반 추론 시작 ---")
    preds = []
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="RAG 추론 진행"):
        question = row['Question']
        
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        if is_multiple_choice(question):
            prompt = f"### 지시:\n당신은 금융보안 전문가입니다. 아래 '법률 내용'을 근거로 '질문'에 가장 적절한 선택지의 '숫자'만 답변하세요.\n\n### 법률 내용:\n{context}\n\n### 질문:\n{question}\n\n### 답변:"
        else:
            prompt = f"### 지시:\n당신은 금융보안 전문가입니다. 아래 '법률 내용'을 근거로 '질문'에 대해 핵심만 간결하게 설명하세요. 근거가 없다면 '정보 없음'이라고 답변하세요.\n\n### 법률 내용:\n{context}\n\n### 질문:\n{question}\n\n### 답변:"
        
        answer_text = llm.invoke(prompt)
        
        final_answer = extract_answer_only(
            generated_text=f"답변:{answer_text}",
            original_question=question,
            prompt=prompt 
        )
        preds.append(final_answer)

    print("--- ✅ RAG 기반 추론 완료 ---\n")

    # 4. 제출 파일 생성
    print("--- 4. 제출 파일 생성 시작 ---")
    OUTPUT_PATH = "results/"
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    submission_df = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
    submission_df['Answer'] = preds
    submission_df.to_csv(os.path.join(OUTPUT_PATH, "rag_pdf_solar.csv"), index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {os.path.join(OUTPUT_PATH, 'rag_pdf_solar.csv')}")
    print("--- ✅ 모든 작업 완료 ---")


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
