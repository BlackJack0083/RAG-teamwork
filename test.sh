#!/bin/bash

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