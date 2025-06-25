# First DREAM Target 2035 Drug Discovery Challenge

This repository contains the complete submission materials for both Step 1 and Step 2 of the **DREAM-SGC Target 2035 Drug Discovery Challenge**, contributed by **Dr. Divya Karade** (Team: TesseractalGene). The challenge focused on using machine learning to identify potential binders for the protein target **WDR91** using either chemical fingerprints (Step 1) or molecular sequences (Step 2).

## 🧠 Challenge Overview

**Organizers:** Structural Genomics Consortium (SGC), DREAM Challenges  
**Target:** WDR91  
**Task:**
- **Step 1:** Predict true actives from DEL screening using ML models on chemical fingerprints.
- **Step 2:** Predict binding affinity (pKd) using a pretrained Transformer-based multi-modal model.

---

## 📁 Repository Structure

```
First-DREAM-Target-2035-Drug-Discovery-Challenge/
│
├── Step1_Fingerprint_Model/
│   ├── TeamTesseractalGene_Step1_Model1.csv             # Prediction file 1
│   ├── TeamTesseractalGene_Step1_Model2.csv             # Prediction file 2
│   ├── TeamTesseractalGene_Step1_Model3.csv             # Prediction file 3
│   ├── Step1_sourcecode-TesseractalGene.py              # Full model training and inference code
│   └── step1_writeup.md                                 # Technical write-up for Step 1
│
├── Step2_MAMMAL_Inference/
│   ├── TeamTesseractalGene_Step2_Model1.csv             # Prediction file 1
│   ├── TeamTesseractalGene_Step2_Model2.csv             # Prediction file 2
│   ├── TeamTesseractalGene_Step2_Model3.csv             # Prediction file 3
│   ├── TesseractalGene_Step2_Workflow.ipynb             # End-to-end MAMMAL inference notebook
│   ├── Preprocessing Code for DREAM Step 2 Test Set.py  # Preprocessing script for Step 2 data
│   ├── code_flow_diagram_step2.png                      # Visual of code logic for Step 2                 
│   └── step2_writeup.md                                 # Technical write-up for Step 2
│
├── LICENSE                                               # License file (optional)
└── README.md                                             # Project overview (this file)
```

---

## 🧪 Step 1: ML Model Using Chemical Fingerprints

- **Input:** ECFP4 fingerprints from ~330K DEL library molecules.
- **Model:** Deep neural network with outlier filtering (Isolation Forest + PCA) and standardization.
- **Evaluation:**
  - Train Accuracy: 97.58%
  - Test Accuracy: 97.50%
  - ROC-AUC: 0.9483
- **Submission Format:** Ranked ~339K molecules with binary labels for top 200 and 500 putative binders.

📄 [Write-up](Step1_Fingerprint_Model/step1_writeup.md) | 🧠 [Training Code](Step1_Fingerprint_Model/step1_model_training_and_inference.py)

---

## 🔬 Step 2: Binding Affinity Prediction Using MAMMAL

- **Input:** Canonical SMILES + WDR91 amino acid sequence
- **Model:** `ibm/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd` from IBM's biomed-multi-alignment framework
- **Environment:** Google Colab + T4 GPU with resume-safe inference
- **Preprocessing:** Deduplication, standardization, Lipinski filtering, normalization
- **Output:** pKd scores (standardized → denormalized → min-max scaled to [0,1])
- **Submission Format:** Ranked predictions for ~339K molecules with top 50 selected.

📄 [Write-up](Step2_MAMMAL_Inference/step2_writeup.md) | 💻 [Colab Pipeline](Step2_MAMMAL_Inference/mammal_inference_colab_pipeline.ipynb)

---

## 🔗 Resources & References

- DREAM-SGC Target 2035 Challenge: https://www.synapse.org/#!Synapse:syn65660836
- IBM MAMMAL Model: https://github.com/BiomedSciAI/biomed-multi-alignment
- Lipinski Rule of Five: Lipinski et al. (2001), Adv. Drug Delivery Reviews

---

## 👩‍🔬 Author

**Dr. Divya Karade**  
Independent Researcher  
Team: TesseractalGene  
✉️ [divya.karade@gmail.com](mailto:divya.karade@gmail.com)

---

## 📜 License

This repository is shared for research and educational purposes in line with the DREAM Challenge’s community spirit. Please cite appropriately if using any component.




