# HotPotQA-LLM
 HotPotQA LLM Question Answering with RAG  

# Data

This repository does not include the full HotPotQA dataset.

Download the dataset from the official HotPotQA website:

```text
https://hotpotqa.github.io/

Required file for this project:

hotpot_dev_fullwiki_v1.json

For the provided Colab notebook, place the file in Google Drive at:

/content/drive/MyDrive/hotpot/hotpot_dev_fullwiki_v1.json

Alternatively, edit DATA_PATH in the notebook to match your local or Drive path.

The full dataset is excluded from GitHub because it is large and should be obtained from the official source.



---

# 6) `results/results.csv` — copy/paste this

```csv
stage,method,em,f1,precision,recall,notes
Stage 1,Question-only prompting,0.016205,0.131470,0.098342,0.263180,Baseline with no evidence/context
Stage 2,Context + strict FINAL output + cleaning,0.207562,0.326949,0.328006,0.382556,Evidence-conditioned prompting with answer cleaning
Stage 3,Embedding + FAISS RAG,0.204727,0.340807,0.337944,0.415472,Best F1; sentence-level retrieval over provided context
Ablation,LLM reranking after FAISS,0.164619,0.283865,0.280844,0.350334,More complex but reduced performance
Ablation,RAG + verification reasoning,0.141256,0.281548,0.267933,0.393947,Verification over-corrected and reduced performance
