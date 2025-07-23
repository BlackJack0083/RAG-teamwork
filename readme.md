# RAG Pipeline for Document Q\&A


This project implements a flexible and extensible Retrieval Augmented Generation (RAG) pipeline designed for question-answering over custom PDF documents. It incorporates advanced retrieval techniques such as **BM25**, **Vector Search**, and **Ensemble Retrieval** with an optional **Cross-Encoder Reranker** to enhance the relevance of retrieved contexts. The pipeline also includes a robust **evaluation framework** to measure performance using BLEU, ROUGE, and reference accuracy metrics.

-----

## ✨ Features

  * **Configurable RAG Pipeline**: Easily customize LLM, embedding models, chunking strategies, and retrieval parameters via command-line arguments.
  * **Hybrid Retrieval**: Combines BM25 (sparse retrieval) and Vector Search (dense retrieval) using an Ensemble Retriever for comprehensive document recall.
  * **Context Reranking**: Integrates a Cross-Encoder Reranker (e.g., `BAAI/bge-reranker-base` or `bge-reranker-v2-m3`) to re-rank retrieved documents, prioritizing the most relevant chunks for the LLM.
  * **Persistent Vector Store**: Utilizes ChromaDB to store document embeddings, allowing for efficient reloading and avoiding redundant embedding generation.
  * **Comprehensive Evaluation**: Automatically evaluates the RAG pipeline's performance using:
      * **BLEU Score**: Measures the fluency and adequacy of generated answers against gold standards.
      * **ROUGE-1 & ROUGE-2 Recall**: Evaluates content overlap (recall) between generated and reference answers.
      * **Reference Accuracy**: Checks if the retrieved context's source page matches the gold reference page.
  * **Jieba Integration**: Supports Chinese text processing by integrating Jieba for tokenization in document loading, splitting, and NLP evaluation metrics.
  * **Detailed Logging**: Saves hyperparameters and evaluation results for each run in a timestamped JSON file, facilitating experiment tracking and comparison.

-----

## 🚀 Getting Started

Follow these steps to set up and run the RAG pipeline.

### Prerequisites

  * Python 3.9+
  * `pip` package manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-rag-project.git
    cd your-rag-project
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK data:**
    Run the following in your Python environment or add it to your setup script:
    ```python
    import nltk
    nltk.download('punkt') # For tokenization in BLEU
    ```

### Configuration

1.  **Environment Variables**:
    Create a `.env` file in the root directory to store your OpenAI API key and base URL (if using a self-hosted or different endpoint for `ChatOpenAI`).

    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    OPENAI_BASE_URL="https://your_llm_api_base_url_here/v1" # e.g., for self-hosted LLM
    ```

    If you're using OpenAI's default API, `OPENAI_BASE_URL` might not be strictly necessary, but it's good practice to include it if your `ChatOpenAI` setup relies on it.

2.  **Prepare your data**:

      * Place your PDF document in the project directory. The default is `./汽车介绍手册.pdf`.
      * Create a JSON file with your QA pairs for evaluation. The default is `./QA_pairs.json`. The format should be a list of dictionaries, like this:
        ```json
        [
          {
            "question": "操作多媒体娱乐功能需要注意什么？",
            "answer": "确保将车辆停驻在安全地点，将挡位切换至驻车挡（P）并使用驻车制动",
            "reference": "page_5"
          },
          {
            "question": "Lynk & Co App 要多少钱？",
            "answer": "结合给定的资料，无法回答问题。",
            "reference": "page_unknown"
          }
        ]
        ```
          * `question`: The query to be posed to the RAG system.
          * `answer`: The gold standard answer for the question.
          * `reference`: The page number(s) where the answer can be found in the original PDF (e.g., "page\_5", "page\_10,page\_12"). Use "page\_unknown" if the answer is not in the document.

### Running the Pipeline

You can run the `main.py` script directly from the command line, customizing parameters as needed. A `test.sh` script is provided for convenience.

#### Using `test.sh` (Recommended for default runs)

```bash
bash test.sh
```

This script will execute `main.py` with predefined arguments, as shown in your `test.sh`:

```bash
python main.py \
    --pdf_path "./汽车介绍手册.pdf" \
    --questions_file "./QA_pairs.json" \
    --llm_model "c101-qwen25-72b" \
    --embedding_model "Qwen3-Embedding-4B" \
    --chunk_size 300 \
    --chunk_overlap 100 \
    --bm25_k 3 \
    --vector_k 3 \
    --ensemble_weights "0.5,0.5" \
    --reranker_model "bge-reranker-v2-m3" \
    --reranker_top_n 3
```

#### Manual Execution with Custom Arguments

You can also run `main.py` directly and specify your own parameters:

```bash
python main.py \
    --pdf_path "./your_document.pdf" \
    --questions_file "./your_qa_data.json" \
    --llm_model "gpt-3.5-turbo" \
    --embedding_model "all-MiniLM-L6-v2" \
    --chunk_size 500 \
    --chunk_overlap 50 \
    --bm25_k 5 \
    --vector_k 10 \
    --ensemble_weights "0.3,0.7" \
    --reranker_model "cross-encoder/ms-marco-TinyBERT-L-2" \
    --reranker_top_n 5
```

-----

## 📁 Project Structure

```
.
├── config.py             # Configuration class for hyperparameters
├── rag_component.py      # Core RAG pipeline components (document loading, splitting, vector store, retrievers, RAG chain)
├── evaluator.py          # Evaluation logic (batch answering, NLP metrics, reference accuracy, result saving)
├── main.py               # Main entry point, orchestrates the RAG pipeline and evaluation
├── test.sh               # Example shell script to run the pipeline with specific arguments
├── .env                  # Environment variables (e.g., API keys - add to .gitignore)
├── requirements.txt      # Python dependencies
├── QA_pairs.json         # Example QA pairs for evaluation
├── 汽车介绍手册.pdf      # Example PDF document
├── chroma_db/            # Directory for persistent ChromaDB vector stores (automatically generated)
│   └── ...
└── rag_results/          # Directory for evaluation results (detailed answers and summary JSONs)
    └── summary_YYYYMMDD_HHMMSS.json
    └── answers_detailed_YYYYMMDD_HHMMSS.json
```

-----

## ⚙️ Configuration (`config.py`)

The `Config` class centralizes all hyperparameters, making it easy to manage and experiment with different settings.

  * **LLM\_MODEL\_NAME**: Name of the large language model to use (e.g., `c101-qwen25-72b`, `gpt-4`).
  * **EMBEDDINGS\_MODEL\_NAME**: Name of the HuggingFace embedding model (e.g., `m3e-base`, `Qwen3-Embedding-4B`).
  * **CHUNK\_SIZE**: Maximum size of text chunks after document splitting.
  * **CHUNK\_OVERLAP**: Overlap between consecutive text chunks.
  * **PDF\_FILE\_PATH**: Path to the input PDF document.
  * **PERSIST\_DIRECTORY**: Directory where ChromaDB will store vector embeddings. Dynamically generated based on embedding model, chunk size, and overlap.
  * **BM25\_K**: Number of documents to retrieve using BM25.
  * **VECTOR\_K**: Number of documents to retrieve using Vector Search.
  * **ENSEMBLE\_WEIGHTS**: Weights for the Ensemble Retriever, e.g., `[0.5, 0.5]` for equal weighting.
  * **RERANKER\_MODEL\_PATH**: Path or name of the Cross-Encoder reranker model (e.g., `BAAI/bge-reranker-base`).
  * **RERANKER\_TOP\_N**: Number of top documents to keep after reranking.
  * **QUESTIONS\_FILE**: Path to the JSON file containing evaluation questions and gold answers.
  * **ANSWERS\_OUTPUT\_FILE**: Temporary path for detailed answers. The actual output files in `rag_results/` will be timestamped.

-----

## 📊 Evaluation Results

After each run, the system will output the evaluation results to the console and save detailed and summary JSON files in the `rag_results/` directory.

### Example Output

```
--- RAG 管道配置 ---
   LLM_MODEL_NAME: c101-qwen25-72b
   EMBEDDINGS_MODEL_NAME: Qwen3-Embedding-4B
   CHUNK_SIZE: 300
   CHUNK_OVERLAP: 100
   PDF_FILE_PATH: ./汽车介绍手册.pdf
   PERSIST_DIRECTORY: ./chroma_db/Qwen3-Embedding-4B_cs300_co100
   BM25_K: 3
   VECTOR_K: 3
   ENSEMBLE_WEIGHTS: [0.5, 0.5]
   RERANKER_MODEL_PATH: bge-reranker-v2-m3
   RERANKER_TOP_N: 3
   QUESTIONS_FILE: ./QA_pairs.json
   ANSWERS_OUTPUT_FILE: ./answers_temp.json
----------------------------------
Loaded 15 pages from 汽车介绍手册.pdf
Split documents into 78 chunks with chunk_size=300, overlap=100.
Creating and persisting new vector store to ./chroma_db/Qwen3-Embedding-4B_cs300_co100
Setting up retrievers...
  BM25 Retriever initialized with k=3
  Vector Retriever initialized with k=3
  Ensemble Retriever initialized with weights=[0.5, 0.5]
  Initializing Reranker model: bge-reranker-v2-m3
Retriever with reranker set up. Top N after reranking: 3
RAG chain set up.

Starting batch answering for 20 questions...
  Processing question 1/20: 操作多媒体娱乐功能需要注意什么？...
  Processing question 10/20: 什么是主动安全？...
  Processing question 20/20: 如何开启巡航控制？...
Batch answering complete. Results saved to rag_results/answers_detailed_20240723_103045.json

--- Evaluation Results ---
平均 BLEU 分数: 0.2543
平均 ROUGE-1 Recall 分数: 0.3876
平均 ROUGE-2 Recall 分数: 0.1522
引用页码命中率: 85.00%

Run summary and hyperparameters saved to: rag_results/summary_20240723_103045.json
```

The `rag_results/` directory will contain JSON files like:

  * `summary_YYYYMMDD_HHMMSS.json`: Contains the hyperparameters used for the run and the aggregated evaluation scores (BLEU, ROUGE-1 Recall, ROUGE-2 Recall, Reference Accuracy).
  * `answers_detailed_YYYYMMDD_HHMMSS.json`: Contains each question, its gold answer, the LLM's generated answer, and the retrieved reference page, allowing for detailed analysis.

-----

## 🤝 Contributing

Contributions are welcome\! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

-----

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

## 🙏 Acknowledgements

  * [LangChain](https://www.langchain.com/) for providing the framework for building LLM applications.
  * [Hugging Face Transformers](https://huggingface.co/transformers/) for access to various pre-trained models.
  * [Chroma](https://www.trychroma.com/) for the vector database.
  * [Jieba](https://github.com/fxsjy/jieba) for Chinese text segmentation.
  * [ROUGE-Chinese](https://www.google.com/search?q=https://github.com/nlpyang/rouge-chinese) for Chinese ROUGE evaluation.

-----

This README provides a comprehensive overview, setup instructions, and details about your project, making it easy for others to understand, use, and contribute to your RAG pipeline.