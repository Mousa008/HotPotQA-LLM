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
