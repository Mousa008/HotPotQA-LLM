# Data

This repository does not include the full HotPotQA dataset.

The dataset should be downloaded from the official HotPotQA website:

```text
https://hotpotqa.github.io/
```

Required file for this project:

```text
hotpot_dev_fullwiki_v1.json
```

For the provided Google Colab notebook, place the file in Google Drive at:

```text
/content/drive/MyDrive/hotpot/hotpot_dev_fullwiki_v1.json
```

Alternatively, edit `DATA_PATH` in the notebook to match your own local or Google Drive path.

---

## Why the Dataset Is Not Included

The full HotPotQA dataset is excluded from this repository because it is large and should be obtained from the official source.

This repository only includes:

- code,
- notebook,
- results summary,
- documentation,
- instructions for downloading and placing the dataset.

---

## Expected File

After downloading the dataset, the expected file is:

```text
hotpot_dev_fullwiki_v1.json
```

Do not commit this file to GitHub.

The `.gitignore` file should exclude it using:

```gitignore
data/hotpot_dev_fullwiki_v1.json
```
