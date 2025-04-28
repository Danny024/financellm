import os
import json
import csv
import time
import math
import hashlib
import re 
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Set, Any


from rag_pipeline import setup_vector_store, create_rag_chain 
from data_processor import process_json_to_markdown, load_markdown_document, get_markdown_splits



def extract_entry_id_from_metadata(metadata: Dict[str, Any]) -> str | None:
    """Extracts the entry ID from the 'Header 1' metadata."""
    if not metadata or "Header 1" not in metadata:
        return None
    header1_content = metadata["Header 1"]
    # Regex to capture the ID part before potential parenthesis
    match = re.search(r"Entry:\s*([^(\s]+)", header1_content) 
    return match.group(1) if match else None

def get_doc_identifier(doc) -> str:
    """Creates a unique identifier for a Langchain Document based on content hash."""
    # Assuming doc has page_content attribute
    if not hasattr(doc, 'page_content'):
         # Handle cases where doc might not be a standard Document object
         # This might happen if the retriever returns something unexpected
         print(f"Warning: Object of type {type(doc)} lacks 'page_content'. Using hash of representation.")
         return hashlib.sha256(repr(doc).encode('utf-8')).hexdigest()
    return hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()


def build_relevant_docs_map(chunks_list: List[Any]) -> Dict[str, Set[str]]:
    """
    Builds a map from entry_id to a set of unique identifiers for chunks 
    belonging to that entry.
    Assumes chunks have metadata and page_content attributes.
    """
    print("Building map of relevant documents for recall calculation...")
    relevant_docs_map = defaultdict(set)
    processed_chunks = 0
    unique_doc_ids = set()
    if not chunks_list:
         print("Warning: No chunks provided to build relevant docs map.")
         return dict(relevant_docs_map) # Return empty dict
         
    for i, chunk in enumerate(chunks_list):
        # Ensure chunk has necessary attributes
        if not hasattr(chunk, 'metadata') or not hasattr(chunk, 'page_content'):
            # print(f"Warning: Chunk {i} is missing metadata or page_content. Skipping.")
            continue
            
        entry_id = extract_entry_id_from_metadata(chunk.metadata)
        if entry_id:
            doc_id = get_doc_identifier(chunk)
            # Add doc_id only if it hasn't been seen before for this entry_id
            if doc_id not in relevant_docs_map[entry_id]:
                 relevant_docs_map[entry_id].add(doc_id)
                 # Track overall unique docs processed correctly
                 if doc_id not in unique_doc_ids:
                      unique_doc_ids.add(doc_id)
                      processed_chunks += 1
            
    print(f"Built map for {len(relevant_docs_map)} unique entry IDs from {processed_chunks} unique chunks.")
    if processed_chunks < len(chunks_list):
         # This count might be slightly off if chunks lack metadata/content
         print(f"Info: Some chunks might have been skipped due to missing data or duplicates.") 
    return dict(relevant_docs_map) # Return as regular dict

def calculate_ndcg(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates NDCG@k using binary relevance."""
    dcg = 0.0
    for i in range(min(k, len(retrieved_ids))):
        if retrieved_ids[i] in relevant_ids:
            dcg += 1.0 / math.log2(i + 2) # relevance=1, discount=log2(rank+1)
            
    idcg = 0.0
    num_relevant_to_consider = min(k, len(relevant_ids))
    for i in range(num_relevant_to_consider):
        idcg += 1.0 / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0



def batch_evaluate(
    retriever: Any, # Langchain retriever instance
    chunks_list: List[Any], # The original list of chunks used to build the vector store
    questions: List[Dict], 
    llm_model_name: str,
    max_k_retrieval: int = 20, # Max docs to fetch for retrieval eval
    eval_k_values: List[int] = [1, 5, 10], # Specific k values to report metrics for in detailed CSV
    output_csv_file: str = "evaluation_results.csv", # Detailed per-question results
    mean_metrics_csv_file: str = "mean_retrieval_metrics.csv", # Aggregated mean metrics
    ollama_base_url: str = "http://localhost:11434",
    use_openai: bool = False # Flag from original rag_pipeline
    ) -> pd.DataFrame:
    """
    Evaluate multiple questions using comprehensive metrics. Saves detailed per-question
    results and aggregated mean retrieval metrics to separate CSV files.

    Args:
        retriever: The Langchain retriever instance.
        chunks_list: The list of Langchain Document chunks used for the vector store.
        questions: List of dictionaries, each with 'question', 'answer', 'doc_id'.
        llm_model_name: The name of the LLM to use for generation (passed to create_rag_chain).
        max_k_retrieval: Max number of documents to retrieve for calculating metrics.
        eval_k_values: List of specific 'k' values to include as columns in the detailed CSV.
        output_csv_file: Path to save the detailed evaluation results per question.
        mean_metrics_csv_file: Path to save the aggregated mean retrieval metrics per k.
        ollama_base_url: Base URL for Ollama service.
        use_openai: Flag to potentially use OpenAI models via rag_pipeline.

    Returns:
        pandas.DataFrame: DataFrame containing detailed evaluation results per question.
    """
    print(f"\n--- Starting Batch Evaluation ({len(questions)} questions) ---")
    results_list = []
    all_latencies = []
    
    # Store metrics across all questions for final averaging
    all_average_precisions = [] # For MAP calculation
    batch_precisions_at_k = defaultdict(list)
    batch_recalls_at_k = defaultdict(list)
    batch_hit_rates_at_k = defaultdict(list)
    batch_ndcgs_at_k = defaultdict(list)

    # --- Pre-computation ---
    relevant_docs_map = build_relevant_docs_map(chunks_list)
    if not relevant_docs_map:
        print("Error: Cannot proceed with evaluation without relevant documents map.")
        return pd.DataFrame()

    # --- Setup RAG Chain ---
    try:
        # Pass llm_model_name to create_rag_chain
        rag_chain = create_rag_chain(
            retriever=retriever, 
            base_url=ollama_base_url, 
            use_openai=use_openai,
        )
        if not rag_chain: raise ValueError("RAG chain creation failed.")
    except Exception as e:
        print(f"Fatal Error: Could not create RAG chain. {e}")
        return pd.DataFrame()

    # --- Evaluates Each Question ---
    for i, q in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}...")
        question = q.get("question")
        expected_answer = q.get("answer")
        expected_doc_id = q.get("doc_id") 

        if not all([question, expected_answer, expected_doc_id]):
            print(f"Warning: Skipping question {i+1} due to missing data (question, answer, or doc_id).")
            continue
            
        relevant_doc_ids = relevant_docs_map.get(expected_doc_id)
        if not relevant_doc_ids:
            print(f"Warning: Skipping question {i+1}. No relevant documents found in map for expected_doc_id: {expected_doc_id}")
            continue
        num_relevant_docs_total = len(relevant_doc_ids)

        # --- Metrics Storage for Question ---
        question_results = {
            "question": question,
            "expected_answer": expected_answer,
            "expected_doc_id": expected_doc_id,
            "generated_answer": "[ERROR]",
            "correctness_containment": 0.0,
            "latency": 0.0,
        }
        for k_val in eval_k_values: # Initialize columns for the detailed CSV
            question_results[f"precision@{k_val}"] = 0.0
            question_results[f"recall@{k_val}"] = 0.0
            question_results[f"hit_rate@{k_val}"] = 0.0
            question_results[f"ndcg@{k_val}"] = 0.0
        question_results["average_precision"] = 0.0 

        start_time = time.time()
        retrieved_docs_for_eval = []
        retrieved_doc_ids_for_eval = []
        generated_answer = "[ERROR]"

        try:
            # --- 1. Retrieval for Metrics ---
            # Use a separate retriever instance for evaluation to fetch max_k docs
            # Assumes retriever passed in has access to the vectorstore
            if hasattr(retriever, 'vectorstore'):
                 eval_retriever = retriever.vectorstore.as_retriever(search_kwargs={'k': max_k_retrieval})
                 retrieved_docs_for_eval = eval_retriever.invoke(question) 
                 retrieved_doc_ids_for_eval = [get_doc_identifier(doc) for doc in retrieved_docs_for_eval]
            else:
                 print("Warning: Cannot create separate eval retriever. Using main retriever for metrics.")
                 # Fallback: use the main retriever, results might be limited by its 'k'
                 retrieved_docs_for_eval = retriever.invoke(question)
                 retrieved_doc_ids_for_eval = [get_doc_identifier(doc) for doc in retrieved_docs_for_eval]

            
            # --- 2. Calculate Retrieval Metrics ---
            precision_sum_for_ap = 0.0
            num_retrieved = len(retrieved_doc_ids_for_eval)
            
            # Calculate metrics for all k up to max_k_retrieval
            for k in range(1, max_k_retrieval + 1):
                if k > num_retrieved: break 

                current_retrieved_set = set(retrieved_doc_ids_for_eval[:k])
                relevant_retrieved_at_k_set = relevant_doc_ids.intersection(current_retrieved_set)
                num_relevant_retrieved_at_k = len(relevant_retrieved_at_k_set)

                precision_at_k = num_relevant_retrieved_at_k / k
                recall_at_k = num_relevant_retrieved_at_k / num_relevant_docs_total
                hit_rate_at_k = 1.0 if num_relevant_retrieved_at_k > 0 else 0.0
                ndcg_at_k = calculate_ndcg(retrieved_doc_ids_for_eval, relevant_doc_ids, k)

                # Store batch metrics for averaging later
                batch_precisions_at_k[k].append(precision_at_k)
                batch_recalls_at_k[k].append(recall_at_k)
                batch_hit_rates_at_k[k].append(hit_rate_at_k)
                batch_ndcgs_at_k[k].append(ndcg_at_k)

                # Store metrics for specific k values in the per-question results dict
                if k in eval_k_values:
                    question_results[f"precision@{k}"] = precision_at_k
                    question_results[f"recall@{k}"] = recall_at_k
                    question_results[f"hit_rate@{k}"] = hit_rate_at_k
                    question_results[f"ndcg@{k}"] = ndcg_at_k

                # Accumulate for Average Precision (AP)
                if retrieved_doc_ids_for_eval[k-1] in relevant_doc_ids:
                     precision_sum_for_ap += precision_at_k
            
            # Final AP for this question
            average_precision_query = precision_sum_for_ap / num_relevant_docs_total if num_relevant_docs_total > 0 else 0.0
            all_average_precisions.append(average_precision_query)
            question_results["average_precision"] = average_precision_query

            # --- 3. Generation ---
            generated_answer = rag_chain.invoke(question) 
            question_results["generated_answer"] = generated_answer

            # --- 4. Calculate Correctness ---
            expected_answer_norm = str(expected_answer).strip().lower().replace('%', '')
            generated_answer_norm = generated_answer.strip().lower().replace('%', '')
            is_correct = expected_answer_norm in generated_answer_norm
            question_results["correctness_containment"] = 1.0 if is_correct else 0.0

        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            # Results dict already has error/default values

        finally:
            latency = time.time() - start_time
            all_latencies.append(latency)
            question_results["latency"] = round(latency, 3)
            results_list.append(question_results)

    # --- Calculate and Print Summary Statistics ---
    print("\n--- Evaluation Batch Summary ---")
    if not results_list:
        print("No results to summarize.")
        return pd.DataFrame(), pd.DataFrame() # Return two empty dataframes

    df_detailed = pd.DataFrame.from_records(results_list)

    # Overall Accuracy
    mean_accuracy = df_detailed["correctness_containment"].mean() * 100
    print(f"Overall Accuracy (Containment): {mean_accuracy:.2f}%")
    
    # Overall Latency
    mean_latency = np.mean(all_latencies)
    print(f"Average Latency per Question: {mean_latency:.3f} seconds")

    # Overall Retrieval Metrics
    mean_average_precision = np.mean(all_average_precisions) if all_average_precisions else 0.0
    print(f"Mean Average Precision (MAP): {mean_average_precision:.4f}")

    print("Mean Retrieval Metrics @ k:")
    mean_metrics_data = []
    # Use all calculated k values for the mean metrics summary
    k_values_calculated = sorted(batch_precisions_at_k.keys()) 
    
    for k_val in k_values_calculated:
         mean_p = np.mean(batch_precisions_at_k.get(k_val, [0]))
         mean_r = np.mean(batch_recalls_at_k.get(k_val, [0]))
         mean_hr = np.mean(batch_hit_rates_at_k.get(k_val, [0]))
         mean_ndcg = np.mean(batch_ndcgs_at_k.get(k_val, [0]))
         
         # Print summary for specific k values requested
         if k_val in eval_k_values or k_val==1 or k_val==max_k_retrieval: # Always print k=1 and max_k
             print(f"  k={k_val}: Recall={mean_r:.4f} | Precision={mean_p:.4f} | HitRate={mean_hr:.4f} | NDCG={mean_ndcg:.4f}")
         
         # Store data for the mean metrics CSV
         mean_metrics_data.append({
             'k': k_val,
             'mean_precision_at_k': mean_p,
             'mean_recall_at_k': mean_r,
             'mean_hit_rate_at_k': mean_hr,
             'mean_ndcg_at_k': mean_ndcg
         })

    df_mean_metrics = pd.DataFrame.from_records(mean_metrics_data)

    # --- Save the Detailed Results to CSV ---
    try:
        output_dir = os.path.dirname(output_csv_file)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        df_detailed.to_csv(output_csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"\nDetailed evaluation results saved to: {output_csv_file}")
    except Exception as e:
        print(f"\nError saving detailed results CSV: {e}")

    # --- Also Save Mean Retrieval Metrics to CSV ---
    try:
        output_dir_mean = os.path.dirname(mean_metrics_csv_file)
        if output_dir_mean: os.makedirs(output_dir_mean, exist_ok=True)
        df_mean_metrics.to_csv(mean_metrics_csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Mean retrieval metrics saved to: {mean_metrics_csv_file}")
    except Exception as e:
        print(f"\nError saving mean retrieval metrics CSV: {e}")

    return df_detailed # Return the detailed dataframe


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Running evaluator script directly...")

    # --- Configuration (Adjust paths relative to project root) ---
    GROUND_TRUTH_FILE = "../data/train.json" 
    MARKDOWN_OUTPUT_FILE = "../data/train_output.md"
    DETAILED_EVAL_OUTPUT_CSV = "evaluation_results.csv" # Detailed per-question results
    MEAN_METRICS_OUTPUT_CSV = "mean_retrieval_metrics.csv" # Mean retrieval metrics per k
    LLM_MODEL_NAME = 'deepseek-r1:1.5b' # Or your chosen base model
    MAX_K_RETRIEVAL = 20 # Max docs to retrieve for eval metrics calculation
    EVAL_K_LIST_DETAILED = [1, 5, 10] # Specific k values to include as columns in detailed CSV
    EVAL_SAMPLE_SIZE = 50 # Number of questions to evaluate from train.json
    RAG_RETRIEVER_K = 5 # How many docs the RAG chain's internal retriever fetches

    # --- Setup the database Phase ---
    vector_store = None
    chunks = []
    retriever = None
    setup_successful = False
    try:
        print("Processing JSON to Markdown...")
        process_json_to_markdown(GROUND_TRUTH_FILE, MARKDOWN_OUTPUT_FILE)
        print("Loading Markdown...")
        markdown_content = load_markdown_document(MARKDOWN_OUTPUT_FILE)
        print("Splitting Markdown...")
        chunks = get_markdown_splits(markdown_content)
        if not chunks: raise ValueError("Markdown splitting resulted in no chunks.")
        print("Setting up Vector Store...")
        vector_store = setup_vector_store(chunks)
        if not vector_store: raise ValueError("Vector store setup failed.")
        
        # This retriever is used by the RAG chain itself
        retriever = vector_store.as_retriever(search_kwargs={'k': RAG_RETRIEVER_K}) 
        print(f"RAG retriever created with k={RAG_RETRIEVER_K}")
        setup_successful = True
    except Exception as e:
        print(f"Error during setup phase: {e}")
        
    # --- Load Questions for Evaluation ---
    evaluation_questions = []
    if setup_successful:
        try:
            with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            
            # Prepare questions in the expected format for batch_evaluate
            for entry in all_data:
                if ("qa" in entry and 
                    "question" in entry["qa"] and 
                    "answer" in entry["qa"] and 
                    "id" in entry and
                    entry["qa"]["question"] is not None and 
                    entry["qa"]["answer"] is not None): 
                    
                    evaluation_questions.append({
                        "question": entry["qa"]["question"],
                        "answer": str(entry["qa"]["answer"]), 
                        "doc_id": str(entry["id"]) 
                    })
            
            if not evaluation_questions:
                 raise ValueError("No valid evaluation questions could be prepared.")
                 
            evaluation_questions = evaluation_questions[:min(EVAL_SAMPLE_SIZE, len(evaluation_questions))]
            print(f"Prepared {len(evaluation_questions)} questions for evaluation.")

        except Exception as e:
             print(f"Error loading or preparing evaluation questions: {e}")
             evaluation_questions = [] 

    # --- Run the Evaluation ---
    if setup_successful and retriever and chunks and evaluation_questions:
        print("Starting evaluation run...")
        evaluation_df = batch_evaluate(
            retriever=retriever, 
            chunks_list=chunks, 
            questions=evaluation_questions,
            llm_model_name=LLM_MODEL_NAME,
            max_k_retrieval=MAX_K_RETRIEVAL,
            eval_k_values=EVAL_K_LIST_DETAILED, # K values for detailed CSV columns
            output_csv_file=DETAILED_EVAL_OUTPUT_CSV,
            mean_metrics_csv_file=MEAN_METRICS_OUTPUT_CSV # Specify output for mean metrics
        )
        print("\nEvaluation script finished.")
        # print(evaluation_df.head()) 
    else:
        print("Evaluation prerequisites not met. Skipping evaluation run.")

