import os
import re
from langchain_community.document_loaders import UnstructuredFileLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import EMBEDDING_MODEL_NAME

def preprocess_text(text: str) -> str:
    """
    텍스트에서 깨진 문자, 특수 기호, 반복되는 패턴 등을 정제합니다.
    """
    # 불필요한 공백과 줄바꿈 정리
    cleaned_text = re.sub(r'\s+', ' ', text)
    # 특수 문자 제거 (예: θ, ㎝ 등) - 일반적인 문자가 아닌 것을 제거
    cleaned_text = re.sub(r'[^\w\s가-힣a-zA-Z0-9.,?!-]', '', cleaned_text)
    # 과도하게 반복되는 하이픈이나 밑줄 제거
    cleaned_text = re.sub(r'-{3,}', ' ', cleaned_text)
    # 한 글자짜리 문자를 제거 (주로 OCR 오류나 잔여물)
    cleaned_text = ' '.join([word for word in cleaned_text.split() if len(word) > 1 or re.match(r'[a-zA-Z0-9]', word)])
    
    return cleaned_text.strip()

def setup_retriever(folder_path="data/laws/"):
    """
    Load documents from a folder and set up the retriever.
    """
    print(f"📄 '{folder_path}' 폴더에서 문서를 로딩하고 벡터화합니다.")
    
    all_documents = []
    
    # 폴더 존재 여부 확인
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == ".pdf":
                print(f"📄 로딩 중: {file_path}")
                # PyPDFLoader 대신 PyMuPDFLoader 사용
                loader = PyMuPDFLoader(file_path)
                documents = loader.load()
                
                # 사전 텍스트 정제 적용
                for doc in documents:
                    doc.page_content = preprocess_text(doc.page_content)
                
                all_documents.extend(documents)
                
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

from sentence_transformers.cross_encoder import CrossEncoder

def rerank_documents(question, documents, k: int = 5):
    """
    RAG 시스템에서 검색된 문서들을 질문과의 관련성을 기준으로 재순위화합니다.
    - question: 사용자 질문
    - documents: 검색 모듈에서 반환된 문서 목록
    - k: 최종적으로 반환할 상위 문서의 개수
    """
    if not documents:
        print("경고: 재순위화할 문서가 없습니다.")
        return []

    print(f"🔍 {len(documents)}개의 문서를 재순위화합니다...")

    # 재순위화 모델 로딩 (CrossEncoder 모델 사용)
    # 이 모델은 질문-문서 쌍의 관련성 점수를 매기는 데 특화되어 있습니다.
    reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2')
    
    # 질문과 각 문서의 내용을 쌍으로 만들어 리스트를 생성
    pairs = [[question, doc.page_content] for doc in documents]
    
    # 모델을 사용하여 각 쌍에 대한 점수를 예측
    scores = reranker.predict(pairs)
    
    # 점수와 문서를 튜플로 묶은 후, 점수 기준으로 내림차순 정렬
    scored_documents = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
    print("✅ 재순위화 완료. 상위 K개 문서 반환.")
    
    # 상위 k개 문서만 반환
    reranked_docs = [doc for score, doc in scored_documents[:k]]
    
    return reranked_docs