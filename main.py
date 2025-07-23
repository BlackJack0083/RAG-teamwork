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
from evaluator import RAGEvaluator
from config import Config
from rag_component import RAGPipeline

# --- Main Execution ---

# --- 主执行流程 ---

def parse_args():
    parser = argparse.ArgumentParser(description="运行一个可配置超参数的 RAG 管道。")

    # 文件路径
    parser.add_argument("--pdf_path", type=str, default="./汽车介绍手册.pdf",
                        help="RAG 所使用的 PDF 文档路径。")
    parser.add_argument("--questions_file", type=str, default="./QA_pairs.json",
                        help="包含评估用 QA 对的 JSON 文件路径。")
    # output_file 参数现在主要是为了用户可见，实际内部会使用带时间戳的文件
    parser.add_argument("--output_file", type=str, default="./answers_temp.json", # 更改默认值以反映其临时性
                        help="详细答案的临时路径。最终结果将带有时间戳。")

    # LLM 和嵌入模型
    parser.add_argument("--llm_model", type=str, default="c101-qwen25-72b",
                        help="要使用的 LLM 模型名称（例如，c101-qwen25-72b）。")
    parser.add_argument("--embedding_model", type=str, default="m3e-base",
                        help="要使用的嵌入模型名称（例如，m3e-base, gte_Qwen2-7B-instruct）。")

    # 文档分割
    parser.add_argument("--chunk_size", type=int, default=300,
                        help="文档分割的文本块大小。")
    parser.add_argument("--chunk_overlap", type=int, default=100,
                        help="文档分割中文本块之间的重叠大小。")

    # 检索器参数
    parser.add_argument("--bm25_k", type=int, default=5,
                        help="BM25 检索器检索的文档数量。")
    parser.add_argument("--vector_k", type=int, default=5,
                        help="向量检索器检索的文档数量。")
    parser.add_argument("--ensemble_weights", type=str, default="0.5,0.5",
                        help="集成检索器的逗号分隔权重（例如，'0.5,0.5'）。")

    # 重排序器参数
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-base",
                        help="Cross-Encoder 重排序模型路径或名称（例如，BAAI/bge-reranker-base）。")
    parser.add_argument("--reranker_top_n", type=int, default=3,
                        help="重排序后保留的前 N 个文档数量。")

    args = parser.parse_args()
    
    # 将 ensemble_weights 字符串转换为浮点数列表
    try:
        args.ensemble_weights = [float(w) for w in args.ensemble_weights.split(',')]
        if len(args.ensemble_weights) != 2 or sum(args.ensemble_weights) == 0:
            raise ValueError("集成权重必须是两个逗号分隔的数字（例如，'0.5,0.5'），并且总和不能为零。")
    except ValueError as e:
        parser.error(f"无效的 --ensemble_weights 格式: {e}。请使用 '0.5,0.5' 等格式。")

    return args


if __name__ == "__main__":
    args = parse_args()
    config = Config(args)

    print("\n--- RAG 管道配置 ---")
    for key, value in config.__dict__.items():
        # 避免打印敏感信息（如果有的话）
        if key not in ['OPENAI_API_KEY', 'OPENAI_BASE_URL']:
            print(f"  {key}: {value}")
    print("----------------------------------")

    # 设置 RAG 管道
    rag_pipeline = RAGPipeline(config)
    
    # 步骤 1: 加载和分割文档
    documents = rag_pipeline.load_documents()
    splits = rag_pipeline.split_documents(documents)

    # 步骤 2: 初始化或加载向量存储
    vectorstore = rag_pipeline.get_vectorstore(splits)

    # 步骤 3: 设置带有重排序的检索器
    retriever = rag_pipeline.setup_retriever(splits)

    # 步骤 4: 设置完整的 RAG 链
    rag_chain = rag_pipeline.setup_rag_chain()

    # 初始化评估器
    # 将 config 直接传递给 RAGEvaluator，它会管理自己的时间戳文件
    evaluator = RAGEvaluator(config) 

    # 步骤 5: 批量回答问题并保存结果（到带时间戳的详细文件）
    evaluator.batch_answer_and_compare(rag_pipeline)

    # 步骤 6: 执行评估并获取总结结果
    evaluation_summary = evaluator.evaluate()

    # 步骤 7: 保存运行详情（超参数和总结结果）
    if evaluation_summary: # 仅当评估成功时才保存
        evaluator.save_run_details(args, evaluation_summary)