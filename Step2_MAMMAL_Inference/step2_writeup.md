#Binding Affinity Prediction for WDR91 Using a Pretrained Multi-modal Transformer (MAMMAL)

Dr. Divya Karade
Independent Researcher / Tesseractal Gene

##Summary Sentence

We used the pretrained MAMMAL model from IBM’s biomed-multi-alignment framework to predict compound binding affinities against WDR91 using a transformer-based multi-modal encoder.

##Background/Introduction

The accurate prediction of drug-target binding affinity (pKd) is a critical step in virtual screening and early-phase drug discovery. In this challenge, we utilised IBM Research’s pretrained MAMMAL model (Multi-modal Alignment Model for Affinity Learning) for predicting pKd values between small molecules and the target protein WDR91.

The motivation for using MAMMAL stems from its architecture, which allows encoding of multiple biological modalities (SMILES, sequences, omics). This makes it ideal for generalising across unseen protein-ligand pairs. Unlike models requiring extensive fine-tuning or docking, MAMMAL enables zero-shot regression leveraging its pretrained Transformer backbone.

We applied this pretrained model directly without fine-tuning, focusing on maximising reproducibility and scalability using Google Colab. To reduce chemical space and enhance drug-likeness, we applied preprocessing steps such as Lipinski filtering, deduplication, normalisation, and standardisation to the input SMILES.

##Methods

###Preprocessing
We started with the test dataset provided in Step 2 and applied the following filters and transformations:
1. Convert parquet files to .CSV file.
1. Deduplication: Removed duplicate SMILES entries.
2. Standardisation: Used RDKit to standardise molecules (canonical SMILES).
3. Normalisation: Adjusted formatting and fixed invalid entries.
4. Lipinski Rule of Five filtering: Ensured drug-likeness of the input set.

###Model Overview
* Model: ibm/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd
* Architecture: Multi-modal Transformer
* Framework: biomed-multi-alignment
* Inputs:
1. Canonical SMILES (small molecules)
2. Amino acid sequence (target protein WDR91)
* Output: Normalised binding affinity score (pKd), later transformed to [0, 1] range using min-max normalisation

###Software Environment and Setup
* All experiments were performed using Google Colab, leveraging T4 GPU when available.
* Resume support: Checkpoints and partial CSV writes prevent data loss
* Required packages were installed via:
```
!pip install -U pip
!pip install biomed-multi-alignment[examples] tdc
```

###Data Handling
* Input file: Filtered-Step2_TestData_Target2035.csv
* Molecules represented as canonical SMILES.
* Protein target: WDR91 (sequence provided by organisers)

###Inference Code Summary
Key steps from the full pipeline run in Colab:
```
# Mount Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')
```

# Imports and model loading
* Used Mammal.from_pretrained() to load the model.
* Used ModularTokenizerOp to tokenize SMILES and protein sequence.
* SMILES and protein sequence were paired into batches (batch size = 3).
* Each batch was:
1. Tokenized
2. Passed through model.forward_encoder_only()
3. Transformed via inverse normalization using provided mean and std
4. Scores were then min-max scaled to the range [0, 1] as required by challenge format.
```
from mammal.model import Mammal
from mammal.examples.dti_bindingdb_kd.task import DtiBindingdbKdTask
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

model = Mammal.from_pretrained("ibm/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd")
model.eval().to(device)
tokenizer_op = ModularTokenizerOp.from_pretrained("ibm/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd")
```

###Prediction Pipeline
* Device Setup:
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

* Resume Support: Partial predictions are resumed from disk to handle Colab interruptions using:
```
if os.path.exists(output_path):
    existing_df = pd.read_csv(output_path)
    done_ids = set(existing_df["RandomID"])
```

* Batch Inference Loop:
```
for batch_df in tqdm(all_batches):
    samples = preprocess_and_tokenize(batch_df)
    outputs = model.forward_encoder_only(samples)
    scores = DtiBindingdbKdTask.process_model_output(outputs)
    append_to_csv(scores)
```

* Batch Size: 3 - 5 (optimised for memory constraints)
* Normalisation:
1. Mean: 5.79384684128215
2. Std: 1.33808027428196
* Output: Normalised pKd score
* Scores were written incrementally to inference_results_partial.csv

###Final Compound Selection
* Final top 50 compounds were selected by sorting scores descending
```
final_df = pd.read_csv(output_path)
final_df = final_df.sort_values(by="Score", ascending=False)
final_df.loc[final_df.head(50).index, "Sel_50"] = 1
final_df.to_csv(final_output_path, index=False)
```

##Compound Ranking and Submission
* All predictions aggregated
* Sorted by Score descending
* Top 50 marked as Sel_50 = 1
* Final CSV format:

| RandomID   | Sel\_50 | Score |
| ---------- | ------- | ----- |
| ID\_XXXXXX | 1       | 6.87  |

* Final binding affinity score (pKd) were scaled using min-max normalisation [0, 1] range to meet submission format requirements.

###Error Handling
* SMILES errors were caught and skipped.
* Runtime interruptions were recoverable via checkpointed output file.

##Code Flow Diagram

${image?fileName=Picture1%2Epng&align=None&scale=50&responsive=true&altText=}

##Conclusion/Discussion

We successfully used a pretrained multi-modal model (MAMMAL) to generate binding affinity scores for ~339K compounds against the WDR91 target. The pipeline was robust, modular, and resilient to GPU fallbacks. While predictions were not fine-tuned, the model generalised well in a zero-shot setting.

Future work could involve comparing fine-tuned variants or integrating protein structure embeddings. The main bottleneck was GPU availability and runtime limits in Colab, which were mitigated using resume and checkpointing logic.

##References
1. IBM Research MAMMAL: https://github.com/BiomedSciAI/biomed-multi-alignment
2. DREAM-SGC Target 2035 Challenge: https://www.synapse.org/Synapse:syn65660836/wiki/
3. Hugging Face: https://huggingface.co/ibm/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd
4. Lipinski, C.A. (2001). Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings.

##Author Statement
Divya Karade: Designed and implemented the pipeline, performed preprocessing, executed inference, and compiled the final submission.

##Source Code

📓 Google Colab Notebook: [Link to Colab Notebook](https://colab.research.google.com/drive/1E1IWSYazZ2JW4Z1aq2O7wO1jRNiZgNEZ?usp=sharing) OR [Link to Github file](https://github.com/DivyaKarade/First-DREAM-Target-2035-Drug-Discovery-Challenge/blob/main/Step2_MAMMAL_Inference/TesseractalGene_Step2_Workflow.ipynb)
📓 Preprocessing Code for DREAM Step 2 Test Set: [Link to Github file](https://github.com/DivyaKarade/First-DREAM-Target-2035-Drug-Discovery-Challenge/blob/main/Step2_MAMMAL_Inference/Preprocessing%20Code%20for%20DREAM%20Step%202%20Test%20Set.py)
