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

# --- 配置类 (超参数) ---
class Config:
    def __init__(self, args):
        # LLM 设置
        self.LLM_MODEL_NAME = args.llm_model
        
        # 嵌入模型
        self.EMBEDDINGS_MODEL_NAME = args.embedding_model

        # 文档处理
        self.CHUNK_SIZE = args.chunk_size
        self.CHUNK_OVERLAP = args.chunk_overlap
        self.PDF_FILE_PATH = args.pdf_path

        # 向量存储
        # 向量存储目录会根据嵌入模型名称动态生成
        self.PERSIST_DIRECTORY = os.path.join(
            "./chroma_db",
            f"{self.EMBEDDINGS_MODEL_NAME.replace('/', '_')}_cs{self.CHUNK_SIZE}_co{self.CHUNK_OVERLAP}"
        )

        # 检索器设置
        self.BM25_K = args.bm25_k
        self.VECTOR_K = args.vector_k
        self.ENSEMBLE_WEIGHTS = args.ensemble_weights # 假定字符串如 "0.5,0.5" 已被解析为列表

        # 重排序器设置
        self.RERANKER_MODEL_PATH = args.reranker_model
        self.RERANKER_TOP_N = args.reranker_top_n

        # 评估文件
        self.QUESTIONS_FILE = args.questions_file
        # 实际的输出文件会由 RAGEvaluator 根据时间戳生成，这里只是为了 argparse 的默认值
        self.ANSWERS_OUTPUT_FILE = args.output_file
