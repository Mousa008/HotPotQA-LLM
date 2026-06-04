# HotPotQA LLM Question Answering with RAG  
### Prompting → Evidence Conditioning → Embedding + FAISS Retrieval

This project builds an end-to-end **LLM-based Question Answering pipeline** for the **HotPotQA** multi-hop question answering dataset.

The system uses **Llama-3.1-8B-Instruct** loaded through **Unsloth 4-bit quantization** in Google Colab. The project progressively improves from a simple question-only prompting baseline to an evidence-grounded **Retrieval-Augmented Generation (RAG)** pipeline.

The final system retrieves relevant evidence from the provided HotPotQA context using **sentence embeddings + FAISS**, then prompts the LLM to generate a short answer span in a strict `FINAL: <answer>` format.

The pipeline includes:

- question-only prompting baseline,
- evidence-conditioned prompting,
- sentence-level retrieval with FAISS,
- answer cleaning and post-processing,
- batching,
- autosave,
- resume support,
- evaluation using the official HotPotQA evaluator.

---

## Project Motivation

HotPotQA is designed for **multi-hop question answering**, where many questions require combining information from multiple Wikipedia passages.

A simple LLM prompt without evidence often guesses from internal model knowledge and performs poorly. This project investigates how adding structured evidence and retrieval improves answer quality.

The project was developed as a practical NLP/LLM coursework pipeline with emphasis on:

- end-to-end reproducibility,
- robust Google Colab execution,
- answer formatting for exact-match evaluation,
- retrieval-based evidence selection,
- experimental comparison across multiple pipeline stages.

---

## Dataset

This project uses the **HotPotQA dev fullwiki dataset**.

Required dataset file:

```text
hotpot_dev_fullwiki_v1.json
```

Source:

```text
https://hotpotqa.github.io/
```

The dataset contains open-domain, multi-hop questions based on Wikipedia passages.

Input fields used:

- `_id`
- `question`
- `context`

Gold fields used only for evaluation:

- `answer`
- `supporting_facts`

The full dataset is **not included** in this repository. See `data/README.md` for download and placement instructions.

---

## Pipeline Overview

The project was implemented in three main stages.

---

### Stage 1 — Question-Only Prompting Baseline

The first baseline prompts the LLM using only the question:

```text
Question: ...
Answer:
```

This tests how well the model answers from internal knowledge alone.

This provides a simple baseline, but it performs poorly because HotPotQA questions usually require evidence from multiple passages.

---

### Stage 2 — Evidence-Conditioned Prompting

The second stage adds evidence from the HotPotQA `context` field.

Instead of asking the model to answer from memory, the prompt includes selected context sentences.

The model is instructed to answer using a strict format:

```text
FINAL: <answer>
```

A post-processing function extracts only the answer span after `FINAL:` and removes common artifacts such as:

- extra explanation,
- parenthetical notes,
- trailing punctuation,
- repeated prompt text,
- template echoes.

This stage significantly improved performance, showing that much of the earlier error came from missing evidence and noisy output formatting.

---

### Stage 3 — Retrieval-Augmented Generation with FAISS

The final selected method uses **Embedding + FAISS RAG**.

For each HotPotQA question:

1. The provided context is flattened into sentence-level candidates.
2. Each candidate sentence is formatted as:

```text
Title: sentence
```

3. Each candidate sentence is embedded using a SentenceTransformers model.
4. The question is embedded using the same model.
5. FAISS retrieves the top-k most relevant evidence sentences using cosine similarity.
6. Retrieved evidence is inserted into the LLM prompt.
7. Llama generates a short answer in the required format:

```text
FINAL: <answer>
```

This method reduces irrelevant context and improves F1 by selecting more relevant evidence before generation.

---

## Best Method

The best final method used:

| Component | Choice |
|---|---|
| Model | Llama-3.1-8B-Instruct |
| Inference framework | Unsloth |
| Quantization | 4-bit |
| Retrieval | SentenceTransformers embeddings + FAISS |
| Prompting | Evidence-conditioned prompt |
| Output format | `FINAL: <answer>` |
| Evaluation | Official HotPotQA evaluator |
| Runtime environment | Google Colab GPU |
| Reliability features | Batching, autosave, resume from saved predictions |

---

## Results

Evaluation was performed on the HotPotQA dev set using the official HotPotQA evaluator.

| Stage | Method | EM | F1 |
|---|---|---:|---:|
| Stage 1 | Question-only prompting | 0.016 | 0.131 |
| Stage 2 | Context + strict `FINAL:` output + cleaning | 0.208 | 0.327 |
| Stage 3 | Embedding + FAISS RAG | 0.205 | 0.341 |

---

## Interpretation

The results show a clear progression.

**Stage 1** performed poorly because the model answered without evidence.

**Stage 2** improved substantially after adding context and enforcing clean answer formatting.

**Stage 3** achieved the best F1 by retrieving more relevant evidence with FAISS before answer generation.

Although Stage 3 EM was slightly lower than Stage 2, its F1 improved. This suggests that retrieval helped produce answers with better token overlap and broader evidence coverage.

---

## Additional Experiments

Two additional advanced variants were tested.

| Experiment | Description | Outcome |
|---|---|---|
| LLM reranking | FAISS retrieve → LLM rerank → answer | Performance decreased |
| Verification reasoning | Retrieve → answer → verify/correct | Performance decreased |

These were useful ablation experiments.

In this setup, the simpler FAISS-only RAG pipeline was more stable and achieved better results than the more complex reasoning and verification variants.

---

## Why RAG Helped

HotPotQA questions often require identifying the correct facts from multiple passages.

Passing too much context can confuse the model, while passing no context forces guessing.

FAISS retrieval helps by selecting a smaller and more relevant evidence set.

The RAG step improved:

- evidence relevance,
- answer grounding,
- recall,
- F1 score,
- robustness compared with question-only prompting.

---

## Repository Structure

```text
hotpotqa-llm-rag/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── notebooks/
│   └── HotPotQA_RAG_Best_Clean_v2_runnable.ipynb
│
├── data/
│   └── README.md
│
└── results/
    ├── results.csv
    └── qualitative_examples.md
```

---

## How to Run

### 1. Open the notebook in Google Colab

Open:

```text
notebooks/HotPotQA_RAG_Best_Clean_v2_runnable.ipynb
```

The notebook is designed for Google Colab.

---

### 2. Download the HotPotQA dataset

Download the HotPotQA dev fullwiki file from the official HotPotQA website:

```text
https://hotpotqa.github.io/
```

Required file:

```text
hotpot_dev_fullwiki_v1.json
```

For the provided Colab notebook, place the file in Google Drive at:

```text
/content/drive/MyDrive/hotpot/hotpot_dev_fullwiki_v1.json
```

Alternatively, edit `DATA_PATH` in the notebook to match your own file location.

---

### 3. Run the notebook cells top-to-bottom

The notebook follows this order:

1. Check GPU.
2. Install dependencies.
3. Mount Google Drive.
4. Load HotPotQA dev data.
5. Load Llama model with Unsloth.
6. Define prompt and answer cleaning functions.
7. Load embedding model.
8. Define FAISS retrieval functions.
9. Generate predictions with autosave/resume.
10. Download the HotPotQA evaluator.
11. Build evaluator-compatible JSON.
12. Run evaluation.

---

## Evaluation

The official HotPotQA evaluator expects prediction files in the following wrapped format:

```json
{
  "answer": {
    "question_id": "predicted answer"
  },
  "sp": {
    "question_id": []
  }
}
```

This project predicts answers only, so supporting facts are left empty:

```json
"sp": {
  "question_id": []
}
```

As a result, supporting-fact metrics and joint metrics are zero.

The main reported metrics are:

- Exact Match,
- F1,
- Precision,
- Recall.

---

## Reproducibility Notes

Important configuration used in the best run:

| Setting | Value |
|---|---|
| Model | Llama-3.1-8B-Instruct |
| Quantization | 4-bit via Unsloth |
| Retrieval | SentenceTransformers + FAISS |
| Generation | Deterministic decoding |
| Output format | `FINAL: <answer>` |
| Storage | Google Drive |
| Reliability | Autosave + resume |

The notebook saves predictions periodically so that Colab disconnects do not lose progress.

---

## Limitations

This project has several limitations:

- It predicts answers only, not supporting facts.
- Retrieval is performed over the provided HotPotQA context, not over the full Wikipedia corpus.
- The model is not fine-tuned.
- Exact Match is sensitive to answer phrasing.
- More complex LLM reranking and verification experiments reduced performance in this setup.
- Results depend on Colab GPU availability and runtime stability.

---

## Future Work

Possible improvements:

- Add supporting-fact prediction.
- Use a stronger embedding model.
- Add cross-encoder reranking instead of LLM reranking.
- Evaluate on a controlled 1,000-question subset for faster experimentation.
- Try multi-hop decomposition before retrieval.
- Convert the notebook into modular Python scripts.
- Add command-line scripts for training, prediction, and evaluation.

---

## Tech Used

- Python
- Google Colab
- Llama-3.1-8B-Instruct
- Unsloth
- Transformers
- SentenceTransformers
- FAISS
- NumPy
- tqdm
- HotPotQA official evaluator

---

## Project Status

Completed experimental pipeline with:

- working baseline,
- context-conditioned prompting,
- RAG with FAISS,
- evaluation results,
- cleaned runnable Colab notebook.

The best-performing final method is the **Embedding + FAISS RAG pipeline**.

---

## Citation / Dataset Credit

HotPotQA dataset:

```bibtex
@inproceedings{yang2018hotpotqa,
  title={{HotpotQA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle={Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  year={2018}
}
```

Dataset and evaluation resources are available from the official HotPotQA website:

```text
https://hotpotqa.github.io/
```
