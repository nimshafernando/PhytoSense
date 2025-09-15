# PhytoSense 🌿  
**Explainable AI-Driven Workflow for Predicting Phytochemicals to Inhibit EGFR in Oral Cancer**

---

## 📌 Project Overview
PhytoSense is my Final Year Project prototype that integrates **computer vision, cheminformatics, and explainable AI** to accelerate drug discovery for oral cancer.  

The system classifies medicinal plant species from leaf images, retrieves their associated phytochemicals, visualizes molecular structures, and sets the foundation for QSAR-based prediction of EGFR inhibition potential.  

This is the **midpoint prototype**, demonstrating:
- Medicinal leaf classification (ensemble deep learning)  
- Phytochemical retrieval from mapped sources  
- 2D/3D molecular visualization via RDKit and py3Dmol  
- Flask web-based prototype with image upload support  

Final implementation will expand to include **QSAR modeling** and **explainability integration**.

---

## 🚀 Features
- **Medicinal Leaf Classification**  
  - Ensemble of ResNet-50, MobileNetV2, EfficientNet-B0  
  - Advanced augmentation (Albumentations), MixUp & CutMix regularization  
  - Ensemble accuracy > 82%  

- **Phytochemical Retrieval**  
  - Linked to ChEMBL, PubChem, and IMPPAT  
  - Returns compound names, SMILES, and PubChem links  

- **2D & 3D Visualization**  
  - RDKit for 2D molecular diagrams  
  - py3Dmol for interactive 3D visualization  

- **Flask Web Prototype**  
  - Upload medicinal plant leaf images  
  - Retrieve phytochemicals and visualize them  
  - Rejects non-medicinal or irrelevant images  

- **Explainability (Planned)**  
  - Gemini 1.5 Flash integration to interpret QSAR results in natural language  

---

## 🛠️ Tech Stack
- **Frontend / UI**: Flask, HTML, CSS  
- **Backend**: Python (PyTorch, Flask)  
- **Models**: ResNet-50, MobileNetV2, EfficientNet-B0 (Ensemble)  
- **Libraries**: Torch, Albumentations, RDKit, py3Dmol, scikit-learn, Matplotlib, Pandas  
- **Dataset**: Mendeley Medicinal Leaf Dataset (80+ species, ~6,900 images)  
- **Environment**: Google Colab Pro (L4 GPU, High-RAM), VS Code for Flask integration  

---


