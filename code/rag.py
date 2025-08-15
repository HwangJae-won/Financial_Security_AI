import os
import pandas as pd
from tqdm import tqdm
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader # PyPDFLoader 추가
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import extract_answer_only, is_multiple_choice
from config import EMBEDDING_MODEL_NAME

def setup_retriever(folder_path="data/laws/"):
    """
    Load documents from a folder and set up the retriever.
    """
    print(f"📄 '{folder_path}' 폴더에서 문서를 로딩하고 벡터화합니다.")
    
    all_documents = []
    
    # 폴더 존재 여부 확인
    if os.path.exists(folder_path):
        # 폴더 내의 모든 파일 순회
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # 파일 확장자에 따라 로더 선택
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == ".pdf":
                print(f"📄 로딩 중: {file_path}")
                loader = PyPDFLoader(file_path)
                all_documents.extend(loader.load())
            elif file_extension == ".txt":
                print(f"📄 로딩 중: {file_path}")
                loader = UnstructuredFileLoader(file_path)
                all_documents.extend(loader.load())
            else:
                print(f"⚠️ 경고: 지원되지 않는 파일 형식 스킵 - {file_path}")

    else:
        raise FileNotFoundError(f"'{folder_path}' 폴더를 찾을 수 없습니다.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(all_documents)
    
    print(f"✨ 임베딩 모델 로딩 중: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("🔍 벡터 스토어(FAISS) 생성 중...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("✅ 벡터 스토어 생성 완료.")
    
    return vectorstore.as_retriever()