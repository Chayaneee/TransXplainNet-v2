# TransXplainNet-v2
TransXplainNet+: A Clinically-Validated LLVM for Chest Radiograph Report Generation


# Introduction: Transformers_Based_Automatic_Report_Generation
This is the official repository of our proposed TransXplainNet+ model details, which is the 2nd version of Fully Transformers_Based_Automatic_Report_Generation. Chest X-ray imaging is crucial for diagnosing and treating thoracic diseases, but the process of examining and generating reports for these images can be challenging. There is a shortage of experienced radiologists, and report generation is time-consuming, reducing effectiveness in clinical settings. To address this issue and advance clinical automation, researchers have been working on automated systems for radiology report generation. However, existing systems have limitations, such as disregarding clinical workflow, ignoring clinical context, and lacking explainability. This paper introduces a novel model for automatic chest X-ray report generation based entirely on transformers. The model focuses on clinical accuracy while improving other text-generation metrics. Our proposed approach, TransXplainNet+, utilizes an off-the-shelf Swin Transformer model along with a transformer-based text encoder that incoporates patient medical history to generate a radiology report. Further, we explore an expert-guided `inside-out' approach and extract only abnormal findings for radiology report refinement. Thus, this study bridges the gap between high-performance automation and the interpretability critical for clinical practice by combining state-of-the-art transformer-based vision encoders, text encoders, and LLMs. The model is trained and tested on a massive X-ray report generation datasets, MIMIC-CXR, demonstrating promising results regarding word overlap, clinical accuracy, and semantic similarity-based metrics. Additionally, qualitative results using Grad-CAM showcase disease location for better understanding by radiologists. The proposed model embraces radiologists' workflow, aiming to improve explainability, transparency, and trustworthiness for their benefit.

# Proposed Pipeline
![Block_Diagram](https://github.com/user-attachments/assets/0100a072-0d41-4387-82c9-e5cf3db32867)


# Data used for Experiments: 

We have used three datasets for this experiment.
  - [MIMIC-CXR](https://physionet.org/content/mimiciii-demo/1.4/)

# Evaluation Metrics 
1. Word Overlap Metrics: BLEU-score, METEOR, ROUGE-L, CIDER
2. Clinical Efficiency (CE) Metrics: AUC, F1-score, Precision, Recall, Accuracy
3. Semantic Similarity-based Metrics: Skip-Thoughts, Average Embedding, Vector Extrema, Greedy Matching
4. Clinical Safety Metrics based on Radiologists Evaluation: Immediate Risk, Long Term Risk, Combined Risk, No Risk
