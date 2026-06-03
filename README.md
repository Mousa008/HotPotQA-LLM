# HotPotQA LLM Question Answering with RAG  
### Prompting → Evidence Conditioning → Embedding + FAISS Retrieval

This project builds an end-to-end **LLM-based Question Answering pipeline** for the **HotPotQA** multi-hop QA dataset. The system uses **Llama-3.1-8B-Instruct** loaded through **Unsloth 4-bit quantization** in Google Colab, and progressively improves from a question-only baseline to an evidence-grounded **Retrieval-Augmented Generation (RAG)** pipeline.

The final system retrieves relevant evidence from the provided HotPotQA context using **sentence embeddings + FAISS**, then prompts the LLM to generate a short answer span in a strict `FINAL: <answer>` format. The pipeline includes batching, autosave, resume support, and evaluation using the official HotPotQA evaluator.

---

## Project Motivation

HotPotQA is designed for **multi-hop question answering**, where many questions require combining information from multiple Wikipedia passages. A simple LLM prompt without evidence often guesses from model memory and performs poorly. This project explores how adding structured evidence and retrieval improves answer quality.

The project was developed as a practical NLP/LLM coursework pipeline with emphasis on:

- end-to-end reproducibility,
- robust Colab execution,
- answer formatting for exact-match evaluation,
- retrieval-based evidence selection,
- experimental comparison across pipeline stages.

---

## Dataset

This project uses the **HotPotQA dev fullwiki dataset**:

- Dataset: `hotpot_dev_fullwiki_v1.json`
- Source: HotPotQA official website  
- Task: open-domain / multi-hop question answering
- Input fields used:
  - `_id`
  - `question`
  - `context`
- Gold fields used only for evaluation:
  - `answer`
  - `supporting_facts`

The full dataset is **not included** in this repository. See `data/README.md` for download and placement instructions.

---

## Pipeline Overview

The project was implemented in three main stages.


### Stage 1 — Question-Only Prompting Baseline

The first baseline prompts the LLM using only the question:

```text
Question: ...
Answer:

This tests how well the model answers from internal knowledge alone. It provides a simple baseline but performs poorly because HotPotQA questions usually require evidence from multiple passages.

Stage 2 — Evidence-Conditioned Prompting

The second stage adds evidence from the HotPotQA context field. Instead of asking the model to answer from memory, the prompt includes selected context sentences.

The model is instructed to answer using a strict format:
FINAL: <answer>

A post-processing function extracts only the answer span after FINAL: and removes common artifacts such as:

extra explanation,
parenthetical notes,
trailing punctuation,
template echoes.

This stage significantly improved performance, showing that much of the earlier error came from both missing evidence and noisy output formatting.

Stage 3 — Retrieval-Augmented Generation with FAISS

The final selected method uses Embedding + FAISS RAG.

For each HotPotQA question:

The provided context is flattened into sentence-level candidates:

Title: sentence
Each candidate sentence is embedded using a SentenceTransformers model.
The question is embedded using the same model.
FAISS retrieves the top-k most relevant evidence sentences using cosine similarity.
Retrieved evidence is inserted into the LLM prompt.
Llama generates a short answer in the required FINAL: <answer> format.

This method reduces irrelevant context and improves F1 by selecting more relevant evidence before generation.

Best Method

The best final method used:

Model: Llama-3.1-8B-Instruct
Inference framework: Unsloth 4-bit quantized loading
Retrieval: SentenceTransformers embeddings + FAISS
Prompting: evidence-conditioned prompt
Output format: FINAL: <answer>
Evaluation: official HotPotQA evaluation script
Runtime environment: Google Colab GPU
Reliability features: batching, autosave, resume from saved predictions
Results

Evaluation was performed on the HotPotQA dev set using the official HotPotQA evaluator.

Variant	Description	EM	F1
Stage 1	Question-only prompting	0.016	0.131
Stage 2	Context + strict FINAL: output + cleaning	0.208	0.327
Stage 3	Embedding + FAISS RAG	0.205	0.341
Interpretation

The results show a clear progression:

Stage 1 performed poorly because the model answered without evidence.
Stage 2 improved substantially after adding context and enforcing clean answer formatting.
Stage 3 achieved the best F1 by retrieving more relevant evidence with FAISS before answer generation.

Although Stage 3 EM was slightly lower than Stage 2, its F1 improved, suggesting that retrieval helped produce answers with better token overlap and broader evidence coverage.

Additional Experiments

Two additional advanced variants were tested:

Experiment	Description	Outcome
LLM reranking	FAISS retrieve → LLM rerank → answer	Performance decreased
Verification reasoning	Retrieve → answer → verify/correct	Performance decreased

These were useful ablation experiments. In this setup, the simpler FAISS-only RAG pipeline was more stable and achieved better results than the more complex reasoning/verification variants.

Why RAG Helped

HotPotQA questions often require identifying the right facts from multiple passages. Passing too much context can confuse the model, while passing no context forces guessing. FAISS retrieval helps by selecting a smaller and more relevant evidence set.

The RAG step improved:

evidence relevance,
answer grounding,
recall,
F1 score,
robustness compared with question-only prompting.
