import os
import re
import pickle
from typing import List, Tuple
from tqdm import tqdm
from transformers import pipeline
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pdfplumber
import faiss

from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter



from text_utils import load_pdf_text, chunk_law_text
from prompt import make_prompt_rag_exaone
from utils import is_multiple_choice, extract_answer_only
from config import MODEL_NAME, EMBEDDING_MODEL_NAME, SCORE_THRESHOLD, TOP_K, FAISS_INDEX_PATH, LAW_PATH


def answer_with_rag(
    question: str,
    vectorstore,
    model,    
    tokenizer,
    top_k: int = 20,
    score_threshold: float = 0.89,
     **generation_params
):
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=top_k)
    
    documents = [doc for doc, score in docs_with_scores]
    scores = [score for doc, score in docs_with_scores]
    
    contexts = [doc.page_content for doc in documents]
    
    if not scores or scores[0] < score_threshold:
        contexts = []
    
    prompt = make_prompt_rag_exaone(
        text=question,
        contexts=contexts,
        use_fewshot=True
    )
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    answer_text = pipe(prompt, **generation_params)
    
    final_answer = extract_answer_only(
        generated_text=f"답변:{answer_text}",
        original_question=question,
        prompt=prompt
    )

    return final_answer

def setup_retriever(folder_path="laws/"):
    print(f"📄 '{folder_path}' 폴더에서 문서를 로딩하고 벡터화합니다.")
    all_documents = []

    if not os.path.exists(FAISS_INDEX_PATH):
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension == ".pdf":
                    print(f"📄 로딩 중: {file_path}")
                    raw_text = load_pdf_text(file_path)
                    print(raw_text[:200])  # --- 디버깅: 추출된 텍스트의 처음 200자 출력 ---
                    if not raw_text.strip():
                        print(f"⚠️ 경고: '{file_path}'에서 텍스트를 추출하지 못했습니다.")
                        continue # 다음 파일로 넘어감

                    chunks = chunk_law_text(raw_text, by_article=True, chunk_size=800, overlap=100)
                    all_documents.extend([Document(page_content=chunk, metadata={'source': file_path}) for chunk in chunks])
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
        print(f"✅ 분할된 문서 수: {len(docs)}")
        if not docs:
            raise ValueError("문서 청크가 생성되지 않았습니다.")
        
        print(f"✨ 임베딩 모델 로딩 중: {EMBEDDING_MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        print("🔍 벡터 스토어(FAISS) 생성 중...")
        vectorstore = LangchainFAISS.from_documents(docs, embeddings)
        print("✅ 벡터 스토어 완료.")
        
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"✅ FAISS 인덱스 저장 완료: {FAISS_INDEX_PATH}")
        
    else:
        print(f"✨ 임베딩 모델 로딩 중: {EMBEDDING_MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = LangchainFAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print(f"✅ FAISS 인덱스 로드 완료: {FAISS_INDEX_PATH}")
        
    return vectorstore