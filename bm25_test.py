import os
import json
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # 导入文本分割器，可选
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import jieba
from langchain_community.retrievers import BM25Retriever

llm_env = dotenv_values(".env")

model = ChatOpenAI(
    model="c101-qwen25-72b",
    base_url=llm_env['OPENAI_BASE_URL'],
    api_key=llm_env['OPENAI_API_KEY']
)
# 初始化嵌入模型
embeddings_model = HuggingFaceEmbeddings(model_name="m3e-small")

# Document loader
# 文档加载器
def load_pdf_with_metadata(file_path):
    """Loads a PDF and returns a list of Document objects with page metadata."""
    loader = PyPDFLoader(file_path=file_path)
    file_name = os.path.basename(file_path)
    loaded_pages = loader.load() # Load the documents (pages)
    total_pages = len(loaded_pages)
    
    for i, page in enumerate(loaded_pages, start=1):
        page_content = page.page_content
        segmented_context = " ".join(jieba.cut(page_content))
        page.page_content = segmented_context
        page.metadata = {
            "file_name": file_name,
            "total_pages": total_pages,
            "page_number": i
        }

    print(f"Loaded {len(loaded_pages)} pages from {file_name}")

    # Return the list of loaded documents
    # 返回加载的文档列表
    return loaded_pages

# --- RAG 管线设置 ---

# 加载 PDF 文档
pdf_file_path = ("./汽车介绍手册.pdf")
document_pages = load_pdf_with_metadata(pdf_file_path)

# 进行文档分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, 
    chunk_overlap=100, 
    length_function=len, 
    is_separator_regex=False
) # Adjust chunk_size and chunk_overlap as needed
splits = text_splitter.split_documents(document_pages)
documents_to_index = splits # Use splits if splitting

# 从文档创建向量存储
# vectorstore = Chroma.from_documents(
#    documents_to_index,
#    embeddings_model,
# )

# 创建一个检索器
# retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

retriever = BM25Retriever.from_documents(documents_to_index, k=1)

# 定义 RAG Chain 的 Prompt 模板
template = """基于以下已知信息，请专业地回答用户的问题。
              不要乱回答，如果无法从已知信息中找到答案，请回答“结合给定的资料，无法回答问题。”。
              诚实地告诉用户。
              结合内容回答，不要输出'#'和'*'等符号，输出内容简洁。
              已知内容：
              {context}
              问题：
              {input}
              """

prompt = ChatPromptTemplate.from_template(template)

rag_chain_from_docs = (
    {
        "input": lambda x: x["input"],
        "context": lambda x: format_docs(x["context"]),
    }
    | prompt
    | model
    | StrOutputParser()
)

retrieve_docs = (lambda x: x["input"]) | retriever

chain = RunnablePassthrough.assign(context=retrieve_docs).assign(answer=rag_chain_from_docs)

def format_docs(docs):
    """将检索到的文档内容拼接成字符串"""
    return "\n\n".join(doc.page_content for doc in docs)

def append_to_json_file(file_path, data):
    """Appends data to an existing JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except(FileNotFoundError, json.JSONDecodeError):
        existing_data = [] # Initialize as empty list if file not found
    if isinstance(existing_data, dict):
        existing_data = [existing_data] # Convert to list if it's a single dictionary
    existing_data.append(data) # Append the new data to the existing data
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

def batch_answer_and_compare(questions_file, output_file):
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    for question in questions:
        # 对问题分词
        segmented_question = " ".join(jieba.cut(question["question"]))
        
        data = {
            "question": question["question"],
            "answer": question["answer"],
            "reference": question["reference"]
        }        
        answers = chain.invoke({"input": segmented_question})
        
        llm_answer = answers["answer"]
        llm_reference = [doc.metadata['page_number'] for doc in answers['context']][0] \
            if answers['context'] else None
        
        data["llm_answer"] = llm_answer
        data["llm_reference"] = f'page_{llm_reference}' if llm_reference else None
        
        print(f"Question: {data['question']}")
        print(f"Reference: {data['reference']}")
        print(f"LLM Answer: {data['llm_answer']}")
        print(f"LLM Reference: {data['llm_reference']}")
        print(f"Answer: {data['answer']}")
        
        append_to_json_file(output_file, data)
        
if __name__ == '__main__':
    batch_answer_and_compare("./QA_pairs.json", "./QA_pairs_answers.json")
        
        