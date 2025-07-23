import os
import json
import argparse
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import jieba
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from sentence_transformers import CrossEncoder
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from config import Config

# --- RAG Components ---

class RAGPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.llm = self._initialize_llm()
        self.embeddings_model = self._initialize_embeddings()
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

    def _initialize_llm(self):
        llm_env = dotenv_values("env")
        if not llm_env.get('OPENAI_BASE_URL') or not llm_env.get('OPENAI_API_KEY'):
            print("Warning: OPENAI_BASE_URL or OPENAI_API_KEY not found in .env. LLM might not initialize correctly.")
        return ChatOpenAI(
            model=self.config.LLM_MODEL_NAME,
            base_url=llm_env.get('OPENAI_BASE_URL'),
            api_key=llm_env.get('OPENAI_API_KEY')
        )

    def _initialize_embeddings(self):
        print(f"Initializing embedding model: {self.config.EMBEDDINGS_MODEL_NAME}")
        return HuggingFaceEmbeddings(model_name=self.config.EMBEDDINGS_MODEL_NAME)

    def load_documents(self):
        """Loads a PDF and returns a list of Document objects with page metadata."""
        if not os.path.exists(self.config.PDF_FILE_PATH):
            raise FileNotFoundError(f"PDF file not found at: {self.config.PDF_FILE_PATH}")
        loader = PyPDFLoader(file_path=self.config.PDF_FILE_PATH)
        file_name = os.path.basename(self.config.PDF_FILE_PATH)
        loaded_pages = loader.load()
        total_pages = len(loaded_pages)
        
        for i, page in enumerate(loaded_pages, start=1):
            page_content = page.page_content
            # Perform Jieba segmentation and store in page_content
            segmented_context = " ".join(jieba.cut(page_content))
            page.page_content = segmented_context
            page.metadata = {
                "file_name": file_name,
                "total_pages": total_pages,
                "page_number": i
            }
        print(f"Loaded {len(loaded_pages)} pages from {file_name}")
        return loaded_pages

    def split_documents(self, documents):
        """Splits documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        )
        splits = text_splitter.split_documents(documents)
        print(f"Split documents into {len(splits)} chunks with chunk_size={self.config.CHUNK_SIZE}, overlap={self.config.CHUNK_OVERLAP}.")
        return splits

    def get_vectorstore(self, documents):
        """
        Checks if the vector store exists locally. If yes, loads it.
        Otherwise, creates a new one and persists it.
        """
        os.makedirs(self.config.PERSIST_DIRECTORY, exist_ok=True)
        if os.path.exists(self.config.PERSIST_DIRECTORY) and len(os.listdir(self.config.PERSIST_DIRECTORY)) > 0:
            print(f"Loading vector store from {self.config.PERSIST_DIRECTORY}")
            vectorstore = Chroma(persist_directory=self.config.PERSIST_DIRECTORY, embedding_function=self.embeddings_model)
        else:
            print(f"Creating and persisting new vector store to {self.config.PERSIST_DIRECTORY}")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings_model,
                persist_directory=self.config.PERSIST_DIRECTORY
            )
        self.vectorstore = vectorstore
        return vectorstore

    def setup_retriever(self, documents_to_index):
        """Sets up the ensemble retriever with optional reranking."""
        print("Setting up retrievers...")
        # 1. Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents_to_index, k=self.config.BM25_K)
        print(f"  BM25 Retriever initialized with k={self.config.BM25_K}")

        # 2. Create vector retriever
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.VECTOR_K})
        print(f"  Vector Retriever initialized with k={self.config.VECTOR_K}")

        # 3. Create EnsembleRetriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=self.config.ENSEMBLE_WEIGHTS
        )
        print(f"  Ensemble Retriever initialized with weights={self.config.ENSEMBLE_WEIGHTS}")

        # 4. Add reranker
        print(f"  Initializing Reranker model: {self.config.RERANKER_MODEL_PATH}")
        try:
            # Use HuggingFaceCrossEncoder for remote models
            reranker_model = HuggingFaceCrossEncoder(
                model_name=self.config.RERANKER_MODEL_PATH,
                model_kwargs={'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES', '') else 'cpu'},
            )
        except Exception as e:
            print(f"Error loading HuggingFaceCrossEncoder: {e}. Trying CrossEncoder (for local paths).")
            # Fallback for local models if HuggingFaceCrossEncoder has issues with local path
            try:
                reranker_model = CrossEncoder(
                    self.config.RERANKER_MODEL_PATH,
                    device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES', '') else 'cpu',
                    # trust_remote_code=True, # might be needed for some specific local models
                    # local_files_only=True # Uncomment if you are sure it's a local path
                )
            except Exception as e_local:
                print(f"Could not load reranker model from {self.config.RERANKER_MODEL_PATH}: {e_local}")
                print("Proceeding without reranker. If you intended to use one, please check the model path and ensure it's accessible.")
                self.retriever = ensemble_retriever # Skip reranker if it fails to load
                return self.retriever

        compressor = CrossEncoderReranker(model=reranker_model, top_n=self.config.RERANKER_TOP_N)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        print(f"Retriever with reranker set up. Top N after reranking: {self.config.RERANKER_TOP_N}")
        return self.retriever

    def setup_rag_chain(self):
        """Defines the RAG chain."""
        template = """基于以下已知信息，请专业地回答用户的问题。
            1. 不要乱回答，如果无法从已知信息中找到答案，请回答“结合给定的资料，无法回答问题。”。
            2. 诚实地告诉用户。
            3. 结合内容回答，不要输出'#'和'*'等符号，输出内容简洁。
            4.  当回答"什么是xxx"的问题时，只回答该术语定义即可。特别的，当询问以下特定术语时，请回答“结合给定的资料，无法回答问题。”，术语包括：“安全出行”、“启动和驾驶”、“驾驶辅助”、“OTA升级”、“Lynk&Co App”、“高压系统”。
            
            example1: 操作多媒体娱乐功能需要注意什么？
            answer1: 确保将车辆停驻在安全地点，将挡位切换至驻车挡（P）并使用驻车制动
              
            example2: Lynk & Co App 要多少钱？
            answer2: 结合给定的资料，无法回答问题。
              
            example3: 如何更换车内的后视镜？
            answer3: 结合给定的资料，无法回答问题。

            example4: 什么是安全出行？
            answer4: 结合给定的资料，无法回答问题。

            已知内容：
            {context}
            问题：
            {input}
            """
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain_from_docs = (
            {
                "input": lambda x: x["input"],
                "context": lambda x: self._format_docs(x["context"]),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        retrieve_docs = (lambda x: x["input"]) | self.retriever
        self.rag_chain = RunnablePassthrough.assign(context=retrieve_docs).assign(answer=rag_chain_from_docs)
        print("RAG chain set up.")
        return self.rag_chain

    def _format_docs(self, docs):
        """Concatenates retrieved document contents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, question: str):
        """Performs a query using the RAG chain."""
        segmented_question = " ".join(jieba.cut(question))
        return self.rag_chain.invoke({"input": segmented_question})