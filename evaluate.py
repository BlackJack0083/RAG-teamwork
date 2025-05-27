import json
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_answers(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_nlp_scores(data):
    rouge = Rouge()
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    
    # For BLEU, we need a smoothing function for cases where there are no common n-grams
    smoothie = SmoothingFunction().method1

    for item in data:
        ref = item["answer"]
        hyp = item.get("llm_answer", "")

        # Compute ROUGE scores
        try:
            scores = rouge.get_scores(hyp, ref, avg=True)
            rouge1_scores.append(scores["rouge-1"]["f"])
            rouge2_scores.append(scores["rouge-2"]["f"])
        except:
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
        
        # Compute BLEU score
        # NLTK's sentence_bleu expects a list of reference sentences and a list of hypothesis tokens
        # We split by space for simplicity, but for more robust BLEU, you might want better tokenization
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        if len(ref_tokens) > 0 and len(hyp_tokens) > 0: # Ensure neither are empty
            bleu_scores.append(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie))
        else:
            bleu_scores.append(0.0) # If either is empty, BLEU score is 0
            
    return bleu_scores, rouge1_scores, rouge2_scores

def compute_reference_accuracy(data):
    correct = 0
    total = 0
    for item in data:
        gold = set(item["reference"].replace("page_", "").split(","))
        pred = set(item.get("llm_reference", "").replace("page_", "").split(","))
        if gold and pred and (gold & pred):  # Ensure both sets are not empty before checking intersection
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0

def main():
    data = load_answers("QA_pairs_answers.json")

    bleu_scores, rouge1_scores, rouge2_scores = compute_nlp_scores(data)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
    
    acc = compute_reference_accuracy(data)

    print(f"平均 BLEU 分数: {avg_bleu:.4f}")
    print(f"平均 ROUGE-1 分数: {avg_rouge1:.4f}")
    print(f"平均 ROUGE-2 分数: {avg_rouge2:.4f}")
    print(f"引用页码命中率: {acc:.2%}")

if __name__ == "__main__":
    main()