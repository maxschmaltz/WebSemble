# WebSemble

WebSemble is a model aiming to use an ensemble approach to solution of [Clickbait Challenge at SemEval 2023](https://pan.webis.de/semeval23/pan23-web/clickbait-challenge.html#evaluation).

This solution was submitted as a solution to the [SemEval 2023 Task 5](https://pan.webis.de/semeval23/pan23-web/clickbait-challenge.html#evaluation) except the LMs were pretrained locally and stored in the folder "models" which is empty here as GitHub cannot store such heavy files. The original dockerized submission includes the pretrained LMS and can be found in DockerHub [maxschmaltz/websemble](https://hub.docker.com/repository/docker/maxschmaltz/websemble/general) with tag _0.28.amd_ or can be pulled with the following command:

```shell
docker pull maxschmaltz/websemble:0.28.amd
```

## How it works
* Summarize (optional);
* Predict label either with summarized text or with its title;
* Retrieve spoiler;
* Postprocess spoiler with predicted label.

## Arguments

| argument                             | description                                                                                                                                                                                                                                                                                                                                                                                                  | required / optional | values                     | default                  |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|----------------------------|--------------------------|
| `input_dir`                          | Directory containing *.jsonl* datasets to be preprocessed. Should obligatory contain *train.jsonl* (*X_train*) and *validation.jsonl* (*X_dev*) if `mode=="train"`, *input.jsonl* (*X_test*) if `mode=="test"`.                                                                                                                                                                                              | required            | any `str`                  | `"./webis22_run"`        |
| `output_dir`                         | Directory to store output in. Output: *run.jsonl* with joint predictions for the subtask and, if `mode=="train"`, *labels.json* and *top_k.json* with labels and top-k spoilers respectively.                                                                                                                                                                                                                | required            | any `str`                  | `"./out"`                |
| `subtask`                            | `"1"` is for the subtask 1 (spoiler classification), `"2"` is for subtask 2 (spoiler retrieval).                                                                                                                                                                                                                                                                                                             | required            | one of `"1"`, `"2"`        | `"2"`                    |
| `-i`, `--instructions_dir`           | Directory containing used models data. Should contain subdirectories */TextClassification* and */QA* with models data for subtasks 1 and 2 respectively.                                                                                                                                                                                                                                                     | optional            | any `str`                  | `"./instructions_local"` |
| `-p`, `--preprocess_mode`            | `"0"`: preprocess *X_train*, *X_dev* and *X_test* (if provided), aim: for initial training (and prediction); `"1"`: preprocess only *X_test* (if provided), aim: for prediction after training; `"2"`: no preprocessing, aim: for evaluation and tests. NB! Preprocess in the context means reading and processing raw data; no preprocessing refers to reading preprocessed previously and saved datasets.  | optional            | one of `"0"`, `"1"`, `"2"` | `"1"`                    |
| `-m`, `--mode`                       | `"train"` forces fine-tuning where applicable, whereas `"test"` skips it and goes directly to prediction.                                                                                                                                                                                                                                                                                                    | optional            | one of `"train"`, `"test"` | `"test"`                 |
| `-s`, `--summarize`                  | Whether to use summarized texts for subtask 1 (spoiler classification); otherwise titles are used.                                                                                                                                                                                                                                                                                                           | optional            | one of `"True"`, `"False"` | `"False"`                |
| `-oc`, `--summarize_only_on_cuda`    | Whether to allow summarization only on CUDA. Ignored if `summarize=="False"`.                                                                                                                                                                                                                                                                                                                                | optional            | one of `"True"`, `"False"` | `"True"`                 |
| `-save`, `--save_datasets`           | Whether to save datasets after preprocessing. Ignored if `summarize=="False"`.                                                                                                                                                                                                                                                                                                                               | optional            | one of `"True"`, `"False"` | `"False"`                |
| `-save_dir`, `--saved_datasets_dir`  | Directory to save datasets after preprocessing to. Ignored if `save_datasets=="False"` or `summarize=="False"`.                                                                                                                                                                                                                                                                                              | optional            | any `str`                  | `"./webis22_summarized"` |

Example usage: `% python3 run.py ./webis22_run ./out 2 -i instructions_local -p 1 -s True -save True -save_dir ./webis22_summarized`

## Models

### Summarization

* [google/pegasus-xsum](https://huggingface.co/google/pegasus-xsum): out of the box

### TextClassification

* [textattack/bert-base-uncased-MNLI](https://huggingface.co/textattack/bert-base-uncased-MNLI): fine-tuned on Webis22 dataset
* [sileod/deberta-v3-base-tasksource-nli](https://huggingface.co/sileod/deberta-v3-base-tasksource-nli): out of the box
* [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased): fine-tuned on Webis22 dataset

### QA

* [facebook/bart-base](https://huggingface.co/facebook/bart-base): fine-tuned on Webis22 dataset
* [bert-large-uncased-whole-word-masking-finetuned-squad](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad): out of the box
* [distilbert-base-cased-distilled-squad](https://huggingface.co/distilbert-base-cased-distilled-squad): out of the box
* [roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2): out of the box

## Execution Time

`% python3 run.py ./webis22_run ./out 2 -m train`

On Apple M1, CPU:

| model                                                 | number of batches | batch size | batches / s | execution time (mm:ss) |
|-------------------------------------------------------|-------------------|------------|-------------|------------------------|
| deberta-v3-base-tasksource-nli                        | 200               | 4          | 7.6         | 00:26                  |
| distilbert-base-uncased                               | 100               | 8          | 14.5        | 00:06                  |
| bert-base-uncased-MNLI                                | 200               | 4          | 10.89       | 00:18                  |
| roberta-base-squad2                                   | 100               | 8          | 0.39        | 04:14                  |
| bert-large-uncased-whole-word-masking-finetuned-squad | 100               | 8          | 0.12        | 13:53                  |
| bart-base                                             | 100               | 8          | 0.12        | 14:15                  |
| distilbert-base-cased-distilled-squad                 | 100               | 8          | 0.88        | 01:52                  |

## Evaluation

Best metrics for the subtasks are below.

| subtask                   | configuration                                     | score      |
|---------------------------|---------------------------------------------------|------------|
| spoiler classification    | 5000 fine-tuning steps + the whole spoiler classification ensemble + the whole spoiler retrieval ensemble | accuracy 0.577          |
| spoiler retrieval         | 5000 fine-tuning steps + DistilBERT base model (uncased) + the whole spoiler retrieval ensemble                                               | BLEU 0.081          |

## Paper

The official paper for the solution is now available on ACL Anthology under https://aclanthology.org/2023.semeval-1.289/:

Maksim Shmalts. 2023. John Boy Walton at SemEval-2023 Task 5: An Ensemble Approach to Spoiler Classification and Retrieval for Clickbait Spoiling. _In Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023)_, pages 2100–2106, Toronto, Canada. Association for Computational Linguistics.

## Citing

```bibtex
@inproceedings{shmalts-2023-john,
    title = "John Boy Walton at {S}em{E}val-2023 Task 5: An Ensemble Approach to Spoiler Classification and Retrieval for Clickbait Spoiling",
    author = "Shmalts, Maksim",
    booktitle = "Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.semeval-1.289",
    doi = "10.18653/v1/2023.semeval-1.289",
    pages = "2100--2106",
    abstract = "Clickbait spoiling is a task of generating or retrieving a fairly short text with a purpose to satisfy curiosity of a content consumer without their addressing to the document linked to a clickbait post or headline. In this paper we introduce an ensemble approach to clickbait spoiling task at SemEval-2023. The tasks consists of spoiler classification and retrieval on Webis-Clickbait-22 dataset. We show that such an ensemble solution is quite successful at classification, whereas it might perform poorly at retrieval with no additional features. In conclusion we outline our thoughts on possible directions to improving the approach and shape a set of suggestions to the said features.",
}
``` 
