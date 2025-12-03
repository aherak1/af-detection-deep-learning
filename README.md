# Detection of Arrhythmias from Nonlinear Features of ECG Signals Using Deep Neural Networks

**Master Thesis**  
**Student:** Ajla Herak  
**Mentor:** R. Prof. Dr. Dušanka Bošković, Dipl. Ing. El.  
**Sarajevo, September 2025**

---

## Abstract

This study addresses the development and evaluation of methods for automatic classification of cardiac rhythms from short single-lead ECG recordings, with a focus on nonlinear representations. The primary objective was to improve the detection accuracy of atrial fibrillation (AFib), normal rhythms, other pathological conditions, and noisy signals, while maintaining balanced performance across classes.  

Experiments were conducted using Poincaré and Hénon plots, recurrent representations, and multimodal models integrating multiple feature modalities. Results demonstrated:

- Models based solely on **Poincaré images** achieved high overall accuracy and stable performance for dominant classes.  
- **Recurrent models** achieved 73.28% overall accuracy but struggled with noisy signals.  
- **Multimodal approaches** (combining recurrent plots, Hénon maps, and numerical features) significantly improved performance.  
  - Recurrent multimodal model: 92% overall accuracy, 85% macro F1-score, 92% weighted F1-score.  
  - Multimodal Hénon model: 85% overall accuracy, balanced class performance.  
  - Fusion of Hénon maps, recurrent plots, and Poincaré sections achieved **97.75% overall accuracy**, stable training, and robust classification of dominant classes.  

These findings confirm the effectiveness of deep convolutional networks in extracting relevant features from nonlinear and recurrent ECG representations, and highlight the benefit of multimodal integration for generalization and accurate rhythm classification.

**Keywords:** automatic ECG signal classification, atrial fibrillation, Poincaré plot, Hénon plot, recurrent representations, multimodal models, Grad-CAM interpretation

---

## Project Overview

This repository contains code, data processing scripts, and models developed for the automatic classification of ECG signals based on nonlinear and recurrent representations. The main objectives are:

1. Preprocessing and normalization of ECG signals.
2. Generation of **nonlinear representations** (Poincaré plots, Hénon maps, recurrent plots).
3. Training of **deep convolutional neural networks** for arrhythmia classification.
4. Implementation of **multimodal fusion strategies** to integrate complementary features.
5. **Visual interpretation of model decisions** using Grad-CAM.
