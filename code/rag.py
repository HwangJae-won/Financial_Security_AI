import torch
from transformers import AutoModelForCausalLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.document import Document # 명확성을 위해 Document 객체를 직접 import
from langchain.document_loaders import UnstructuredFileLoader
# 더 안정적인 RecursiveCharacterTextSplitter를 사용하도록 변경
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import HuggingFaceEmbeddings, CacheBackedEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# 기존 model.py의 load_model 함수를 그대로 사용합니다.
from model import load_model

class RAGPipeline:
    def __init__(self, embedding_model_name="jhgan/ko-sroberta-multitask", file_path="data/law.txt"):
        """
        RAG 파이프라인을 초기화하고 필요한 모든 구성 요소를 설정합니다.
        """
        print("✅ RAG 파이프라인 초기화를 시작합니다.")
        
        # 1. LLM 로드
        print("🧠 LLM을 로딩합니다...")
        hf_pipeline = load_model()
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        print("✅ LLM 로딩 완료.")

        # 2. 문서 로드 및 분할 (오류 해결을 위해 로직을 명확하게 변경)
        print(f"📄 문서를 로딩합니다: {file_path}")
        loader = UnstructuredFileLoader(file_path)
        # loader.load()는 Document 객체들의 리스트를 반환합니다.
        # 데이터 타입: List[Document]
        raw_documents = loader.load()

        print("📄 문서를 청크 단위로 분할합니다...")
        # 일반 텍스트에 더 안정적인 RecursiveCharacterTextSplitter를 사용합니다.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
        )
        # split_documents 함수는 Document 리스트를 입력받아,
        # 각 Document의 내용(page_content)을 분할한 뒤,
        # 새로운 Document 청크들의 리스트를 반환합니다.
        # 입력 타입: List[Document] -> 출력 타입: List[Document]
        documents = text_splitter.split_documents(raw_documents)
        
        # 분할된 문서가 없는 경우 오류를 발생시켜 문제를 조기에 파악합니다.
        if not documents:
            raise ValueError("문서 분할 후 생성된 청크가 없습니다. law.txt 파일이 비어있는지 확인해주세요.")
        
        print(f"✅ 문서 분할 완료. 총 {len(documents)}개의 조각 생성.")

        # 3. 임베딩 모델 및 캐시 설정
        print(f"✨ 임베딩 모델을 로딩합니다: {embedding_model_name}")
        core_embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        store = LocalFileStore("./cache/embeddings")
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            core_embeddings_model, store, namespace=embedding_model_name
        )
        print("✅ 임베딩 모델 및 캐시 설정 완료.")

        # 4. 벡터 저장소(Vector Store) 및 검색기(Retriever) 생성
        print("🔍 벡터 저장소를 생성합니다...")
        self.vectorstore = FAISS.from_documents(documents, cached_embeddings)
        self.retriever = self.vectorstore.as_retriever()
        print("✅ 벡터 저장소 및 검색기 생성 완료.")

        # 5. 프롬프트 템플릿 정의
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    당신은 금융보안 전문가입니다.
                    반드시 주어진 컨텍스트(Context)만을 사용하여 질문에 답변해야 합니다.
                    컨텍스트에서 답변을 찾을 수 없다면, "정보를 찾을 수 없습니다."라고만 답변하고 절대 내용을 지어내지 마세요.
                    답변은 한국어로, 간결하고 명확하게 작성해주세요.

                    ---
                    Context:
                    {context}
                    ---
                    """
                ),
                ("human", "질문: {question}"),
            ]
        )
        print("✅ 프롬프트 템플릿 설정 완료.")

        # 6. LangChain Expression Language (LCEL)을 사용한 체인 구성
        self.chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
        )
        print("✅ RAG 체인 구성 완료. 파이프라인이 준비되었습니다.")

    def invoke(self, question: str) -> str:
        """
        주어진 질문에 대해 RAG 체인을 실행하고 답변을 반환합니다.
        """
        print(f"\n🚀 '{question}'에 대한 추론을 시작합니다...")
        result = self.chain.invoke(question)
        return result

# 메인 실행 블록 (테스트용)
if __name__ == "__main__":
    rag_pipeline = RAGPipeline()
    test_question = "전자금융거래법의 목적은 무엇인가?"
    answer = rag_pipeline.invoke(test_question)
    
    print("\n\n--- 최종 답변 ---")
    print(answer)
    print("----------------")
