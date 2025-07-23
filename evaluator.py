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
from rouge_chinese import Rouge # Import rouge_chinese
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from config import Config
from rag_component import RAGPipeline

# --- Evaluation Components ---

class RAGEvaluator:
    def __init__(self, config: Config):
        self.config = config
        # Define a unique results file name based on current timestamp or run ID
        timestamp = os.environ.get("RUN_TIMESTAMP", "") # Using environment variable if set by a runner
        if not timestamp:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results in a dedicated 'results' directory
        self.results_dir = "rag_results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.summary_file = os.path.join(self.results_dir, f"summary_{timestamp}.json")
        self.detailed_answers_file = os.path.join(self.results_dir, f"answers_detailed_{timestamp}.json")

    def append_to_json_file(self, file_path, data):
        """Appends data to an existing JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []
        if isinstance(existing_data, dict):
            existing_data = [existing_data]
        existing_data.append(data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

    def batch_answer_and_compare(self, rag_pipeline: RAGPipeline):
        """Processes questions, generates answers, and stores them for evaluation."""
        if not os.path.exists(self.config.QUESTIONS_FILE):
            raise FileNotFoundError(f"Questions file not found at: {self.config.QUESTIONS_FILE}")
            
        with open(self.config.QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        # Use the newly defined detailed_answers_file for this run
        if os.path.exists(self.detailed_answers_file):
            os.remove(self.detailed_answers_file)
            print(f"Cleared existing output file: {self.detailed_answers_file}")

        print(f"\nStarting batch answering for {len(questions)} questions...")
        for i, question_item in enumerate(questions):
            if (i + 1) % 10 == 0 or i == 0 or (i + 1) == len(questions):
                print(f"  Processing question {i+1}/{len(questions)}: {question_item['question'][:50]}...")
            
            data = {
                "question": question_item["question"],
                "answer": question_item["answer"], # Gold answer
                "reference": question_item["reference"] # Gold reference
            }
            
            answers = rag_pipeline.query(question_item["question"])
            
            llm_answer = answers["answer"]
            # Only get the page number from the first reranked document
            llm_reference = (
                f'page_{answers["context"][0].metadata["page_number"]}'
                if answers['context'] and 'page_number' in answers['context'][0].metadata
                else None
            )
            
            data["llm_answer"] = llm_answer
            data["llm_reference"] = llm_reference
            
            self.append_to_json_file(self.detailed_answers_file, data)
        print(f"Batch answering complete. Results saved to {self.detailed_answers_file}")

    def load_answers(self, file_path):
        """Loads the generated answers from a JSON file."""
        if not os.path.exists(file_path):
            print(f"Warning: Output file '{file_path}' not found. No answers to load for evaluation.")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def compute_nlp_scores(self, data):
        """Computes BLEU and ROUGE recall scores."""
        rouge = Rouge() # Use rouge_chinese
        bleu_scores = []
        rouge1_r_scores = []  # Changed to recall
        rouge2_r_scores = []  # Changed to recall
        
        smoothie = SmoothingFunction().method1

        for item in data:
            ref = item["answer"]
            hyp = item.get("llm_answer", "")

            # Word segmentation for Chinese text using jieba
            ref_seg_tokens = list(jieba.cut(ref)) # Get tokens as a list directly
            hyp_seg_tokens = list(jieba.cut(hyp)) # Get tokens as a list directly
            
            # For ROUGE, join tokens with space
            ref_seg_for_rouge = ' '.join(ref_seg_tokens)
            hyp_seg_for_rouge = ' '.join(hyp_seg_tokens)
            
            try:
                scores = rouge.get_scores(hyp_seg_for_rouge, ref_seg_for_rouge) # Pass segmented text
                rouge1_r_scores.append(scores[0]["rouge-1"]["r"])  # Appending recall score
                rouge2_r_scores.append(scores[0]["rouge-2"]["r"])  # Appending recall score
            except ValueError:
                rouge1_r_scores.append(0.0)
                rouge2_r_scores.append(0.0)
            
            # For BLEU, pass the list of segmented tokens directly
            if len(ref_seg_tokens) > 0 and len(hyp_seg_tokens) > 0:
                bleu_scores.append(sentence_bleu([ref_seg_tokens], hyp_seg_tokens, smoothing_function=smoothie))
            else:
                bleu_scores.append(0.0)
                
        return bleu_scores, rouge1_r_scores, rouge2_r_scores # Return recall scores

    def compute_reference_accuracy(self, data):
        """Computes the accuracy of retrieved reference pages, specifically for single page predictions."""
        correct = 0
        total = 0
        for item in data:
            gold_references = set(item["reference"].replace("page_", "").split(","))
            
            llm_reference_str = item.get("llm_reference", "")
            pred_reference_page = llm_reference_str.replace("page_", "") if llm_reference_str else None
            
            if pred_reference_page and pred_reference_page in gold_references:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0.0

    def evaluate(self):
        """Runs the complete evaluation process and saves results."""
        data = self.load_answers(self.detailed_answers_file)
        if not data:
            print("No data available for evaluation.")
            return {} # Return an empty dict if no data

        bleu_scores, rouge1_r_scores, rouge2_r_scores = self.compute_nlp_scores(data) # Changed variable names
        
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        avg_rouge1_r = sum(rouge1_r_scores) / len(rouge1_r_scores) if rouge1_r_scores else 0.0 # Changed to recall
        avg_rouge2_r = sum(rouge2_r_scores) / len(rouge2_r_scores) if rouge2_r_scores else 0.0 # Changed to recall
        
        ref_accuracy = self.compute_reference_accuracy(data)

        print("\n--- Evaluation Results ---")
        print(f"平均 BLEU 分数: {avg_bleu:.4f}")
        print(f"平均 ROUGE-1 Recall 分数: {avg_rouge1_r:.4f}") # Changed print statement
        print(f"平均 ROUGE-2 Recall 分数: {avg_rouge2_r:.4f}") # Changed print statement
        print(f"引用页码命中率: {ref_accuracy:.2%}")

        # Prepare summary results to save
        summary_results = {
            "timestamp": os.path.basename(self.summary_file).replace("summary_", "").replace(".json", ""),
            "avg_bleu": round(avg_bleu, 4),
            "avg_rouge1_recall": round(avg_rouge1_r, 4), # Changed key
            "avg_rouge2_recall": round(avg_rouge2_r, 4), # Changed key
            "reference_accuracy": round(ref_accuracy, 4),
            "num_questions_evaluated": len(data)
        }
        return summary_results

    def save_run_details(self, config_args, evaluation_summary):
        """Saves the hyperparameters and evaluation summary for the current run."""
        run_data = {
            "hyperparameters": vars(config_args), # Convert argparse Namespace to dict
            "evaluation_results": evaluation_summary
        }
        
        # Save to the unique summary file for this run
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(run_data, f, indent=4, ensure_ascii=False)
            print(f"\nRun summary and hyperparameters saved to: {self.summary_file}")
        except Exception as e:
            print(f"Error saving run summary to {self.summary_file}: {e}")