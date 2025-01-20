# TransXplainNet-v2
TransXplainNet+: A Clinically-Validated LLVM for Chest Radiograph Report Generation


# Introduction: Transformers_Based_Automatic_Report_Generation
This is the official repository of our proposed TransXplainNet+ model details, which is the 2nd version of Fully Transformers_Based_Automatic_Report_Generation. Chest X-ray imaging is crucial for diagnosing and treating thoracic diseases, but the process of examining and generating reports for these images can be challenging. There is a shortage of experienced radiologists, and report generation is time-consuming, reducing effectiveness in clinical settings. To address this issue and advance clinical automation, researchers have been working on automated systems for radiology report generation. However, existing systems have limitations, such as disregarding clinical workflow, ignoring clinical context, and lacking explainability. This paper introduces a novel model for automatic chest X-ray report generation based entirely on transformers. The model focuses on clinical accuracy while improving other text-generation metrics. Our proposed approach, TransXplainNet+, utilizes an off-the-shelf Swin Transformer model along with a transformer-based text encoder that incoporates patient medical history to generate a radiology report. Further, we explore an expert-guided `inside-out' approach and extract only abnormal findings for radiology report refinement. Thus, this study bridges the gap between high-performance automation and the interpretability critical for clinical practice by combining state-of-the-art transformer-based vision encoders, text encoders, and LLMs. The model is trained and tested on a massive X-ray report generation datasets, MIMIC-CXR, demonstrating promising results regarding word overlap, clinical accuracy, and semantic similarity-based metrics. Additionally, qualitative results using Grad-CAM showcase disease location for better understanding by radiologists. The proposed model embraces radiologists' workflow, aiming to improve explainability, transparency, and trustworthiness for their benefit.

# Proposed Pipeline
![Block_Diagram](https://github.com/user-attachments/assets/0100a072-0d41-4387-82c9-e5cf3db32867)


# Data used for Experiments: 

We have used three datasets for this experiment.
  - [MIMIC-CXR](https://physionet.org/content/mimiciii-demo/1.4/)
  - [VinDr-CXR](https://vindr.ai/datasets/cxr)
  - Hospital-Dataset: It's private and confidential.

# Evaluation Metrics 
1. Word Overlap Metrics: BLEU-score, METEOR, ROUGE-L, CIDER
2. Clinical Efficiency (CE) Metrics: AUC, F1-score, Precision, Recall, Accuracy
3. Semantic Similarity-based Metrics: Skip-Thoughts, Average Embedding, Vector Extrema, Greedy Matching
4. Clinical Safety Metrics based on Radiologists Evaluation: Immediate Risk, Long Term Risk, Combined Risk, No Risk

# Quantitative Results (Word-Overlap Metrics)

| **Models**                             | **B1**   | **B2**   | **B3**   | **B4**   | **METEOR** | **ROUGE-L** | **CIDER**    |
|----------------------------------------|----------|----------|----------|----------|------------|-------------|--------------|
| Nooralahzadeh et al. 2021             | 0.378    | 0.232    | 0.154    | 0.107    | 0.145      | 0.272       | --           |
| Liu et al. (PPKED) 2021                         | 0.360    | 0.224    | 0.149    | 0.106    | 0.149      | 0.284       | --           |
| Nguyen et al. 2021                  | **0.495**| **0.360**| **0.278**| **0.224**| **0.222**  | **0.390**   | --           |
| Wang et al. (XPRONET) 2022                        | 0.344    | 0.215    | 0.146    | 0.105    | 0.138      | 0.279       | --           |
| Mondal et al. (TransXplainNet) 2023                | 0.376    | 0.255    | 0.187    | 0.145    | 0.161      | 0.310       | _0.219_      |
| TransXplainNet+                        |$\underline{0.418}$ | $\underline{0.290}$ |$\underline{0.215}$ | $\underline{0.168}$ | $\underline{0.179}$  |$\underline{0.330}$ | **0.279**    |

# Qualititative Results (CE and Semantic Similarity Metrics)

| **Models**                             | **AUC**   | **F1S**  | **Precision** | **Recall**  | **Accuracy** | **ST**    | **AE**    | **VE**    | **GM**    |
|----------------------------------------|-----------|----------|---------------|-------------|--------------|-----------|-----------|-----------|-----------|
| Transformer Prog. 2021              | --        | 0.308    | 0.240         | **0.428**   | --           | --        | --        | --        | --        |
| Co-ATT 2021                        | --        | 0.303    | 0.352         | 0.298       | --           | --        | --        | --        | --        |
| Nguyen et al. 2021                 | **0.784** | **0.412**| $\underline{0.432}$ | $\underline{0.418}$  | **0.887**    | --        | --        | --        | --        |
| CvT-212DistilGPT2 2022             | --        | 0.390    | 0.365         | 0.418       | --           | --        | --        | --        | --        |
| TransXplainNet 2023                | 0.664     | 0.325    | 0.321         | 0.361       | 0.793        | $\underline{0.738}$ | $\underline{0.937}$   |$\underline{0.508}$    | $\underline{0.765}$  |
| TransXplainNet+                        | $\underline{0.721}$ | $\underline{0.393}$ | **0.457**     | 0.395       | $\underline{0.829}$     | **0.744** | **0.945** | **0.533** | **0.781** |

# Qualitative Results
<img width="868" alt="Image" src="https://github.com/user-attachments/assets/a29a4008-878a-4e31-85ed-32d62f346435" />
