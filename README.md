# 🩻 AI-Assisted Chest X-Ray Analysis Using Vision-Language Models in PACS

An AI-assisted **PACS-style chest X-ray analysis workstation** integrating **BiomedCLIP** for vision-language inference and **Qwen2.5-1.5B-Instruct** for structured radiology-style report generation.

The system demonstrates how pretrained multimodal AI models can be integrated into a unified medical imaging workflow covering image upload, preprocessing, AI-assisted prediction, explainability, report generation, study management, and PDF export.

> **Disclaimer:** This project is a research and educational prototype. Its outputs are not clinically validated and must not be used as medical diagnoses.

---

## 📌 Project Overview

Chest X-ray (CXR) interpretation requires careful and consistent examination by trained radiologists. Increasing imaging workloads can make the interpretation and reporting process more demanding.

Although artificial intelligence can support medical image analysis, many existing AI systems operate as isolated prediction tools and provide limited integration with the Picture Archiving and Communication System (PACS) workflow.

This project investigates how a **Vision-Language Model (VLM)** and **Large Language Model (LLM)** can be combined within a PACS-style environment to provide:

* Chest X-ray image analysis
* Ranked disease predictions
* Confidence scores
* Grad-CAM explainability
* Structured radiology-style reports
* Study storage and retrieval
* PDF report export

The project focuses primarily on **system integration, technical functionality, response time, output consistency, and workflow integration**, rather than clinical diagnostic validation.

---

## 🎯 Research Objectives

1. **Integrate pretrained AI models**
   Combine BiomedCLIP and Qwen2.5-1.5B-Instruct to generate descriptive and explainable chest X-ray outputs.

2. **Develop a PACS-style platform**
   Create an integrated environment for image upload, visualization, AI-assisted analysis, report generation, study management, and report export.

3. **Evaluate the developed prototype**
   Assess system functionality, response time, output consistency, error handling, stability, and workflow integration.

---

## 🧠 AI Models

### BiomedCLIP

**BiomedCLIP** serves as the Vision-Language Model.

It performs zero-shot image-text matching between the uploaded chest X-ray and predefined textual descriptions representing thoracic findings.

The model produces:

* Primary predicted finding
* Ranked alternative findings
* Confidence scores
* Visual information used for explainability

No additional model training is performed in this prototype.

### Qwen2.5-1.5B-Instruct

**Qwen2.5-1.5B-Instruct** serves as the Large Language Model responsible for report generation.

BiomedCLIP predictions and confidence information are inserted into a structured prompt before being passed to Qwen2.5.

The generated report contains:

* **Technique**
* **Findings**
* **Impression**

The LLM is used for **inference only** and generates observational, non-diagnostic text requiring professional confirmation.

---

## 🔄 System Workflow

The complete inference workflow is:

```text
Chest X-Ray Upload
        │
        ▼
File Validation
        │
        ▼
Image Preprocessing
Resize to 224 × 224 + Normalization
        │
        ▼
BiomedCLIP
Vision-Language Inference
        │
        ├──────────────► Grad-CAM Explainability
        │
        ▼
Disease Predictions
+ Confidence Scores
        │
        ▼
Prompt Engineering
        │
        ▼
Qwen2.5-1.5B-Instruct
        │
        ▼
Structured Radiology-Style Report
        │
        ▼
MongoDB / GridFS Storage
        │
        ▼
PACS-Style Viewer
        │
        ├── Study History
        ├── Report Editing
        └── PDF Export
```

---

## 🏗️ System Architecture

The prototype follows a modular architecture.

```text
┌───────────────────────────────┐
│      Streamlit Frontend       │
│     PACS-Style Workstation    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│        FastAPI Backend        │
│ API + Processing Coordinator  │
└───────────────┬───────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌──────────────┐  ┌─────────────────┐
│ BiomedCLIP   │  │ Image Processing│
│     VLM      │  │   + Grad-CAM    │
└──────┬───────┘  └─────────────────┘
       │
       ▼
┌───────────────────────────────┐
│      Prompt Engineering       │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│    Qwen2.5-1.5B-Instruct      │
│      Report Generation        │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│       MongoDB + GridFS        │
│ Studies • Reports • Images    │
└───────────────────────────────┘
```

### FastAPI

FastAPI acts as the **backend API and processing coordinator**.

It connects the frontend with the image-processing functions, AI models, and database.

### ngrok

During development and demonstration, **ngrok** can be used as a secure tunneling service to expose the locally running FastAPI backend through a temporary public endpoint.

In simple terms:

```text
Frontend
   │
   ▼
ngrok Public Tunnel
   │
   ▼
Local FastAPI Backend
   │
   ├── BiomedCLIP
   ├── Qwen2.5
   └── MongoDB
```

FastAPI and ngrok therefore have different roles:

* **FastAPI** = backend API / system coordinator
* **ngrok** = secure tunnel between the internet and the local backend

---

## 🖥️ System Modules

### 1. Landing Page

Introduces the workstation and its three main capabilities:

* DICOM Viewer
* AI Triage
* Structured Reports

---

### 2. Worklist

Provides a PACS-style study queue containing:

* Total studies
* Analyzed studies
* Pending studies
* Predicted findings
* Priority levels
* Analysis status
* Search
* Date sorting
* Study opening

---

### 3. Upload

Supports chest X-ray uploads in:

* PNG
* JPG
* JPEG
* DICOM

The uploaded image is validated before being sent through the inference pipeline.

---

### 4. Viewer

The Viewer is the core component of the system.

It provides:

* Chest X-ray visualization
* Window/level adjustment
* Brightness adjustment
* Contrast adjustment
* Image inversion
* AI prediction
* Confidence scores
* Grad-CAM visualization
* Generated report
* Report editing
* PDF export
* Re-analysis
* Study deletion

---

## 🔥 Explainable AI with Grad-CAM

Grad-CAM is incorporated into the viewer to provide visual explainability.

The visualization highlights anatomical image regions that influenced the AI prediction.

```text
Chest X-Ray
     │
     ▼
BiomedCLIP Prediction
     │
     ▼
Grad-CAM
     │
     ▼
Highlighted Influential Regions
```

This provides additional context beyond displaying only a predicted disease label and confidence score.

The heatmap should not, however, be interpreted independently as clinical evidence.

---

## 📝 Structured Report Generation

After BiomedCLIP generates its predictions, the results are converted into a structured prompt.

Example conceptual flow:

```text
BiomedCLIP Prediction
        +
Confidence Information
        │
        ▼
Structured Prompt
        │
        ▼
Qwen2.5-1.5B-Instruct
        │
        ▼
Technique
Findings
Impression
```

Prompt engineering is used to control the structure and wording of the generated report.

This is different from the **inference pipeline**:

* **Prompt engineering** defines the instructions given to the LLM.
* **Inference pipeline** represents the complete process from chest X-ray input to final system output.

---

## ✏️ Human Review

Generated reports can be edited by the user before finalization.

Edits are stored with timestamps to support an audit trail between the AI-generated output and the reviewed version.

This maintains a **human-in-the-loop** approach where the AI assists the interpretation process rather than replacing professional judgment.

---

## 📄 PDF Report Export

The system can generate a formatted PDF containing relevant study information, including:

* Chest X-ray image
* Predicted finding
* Alternative predictions
* Confidence scores
* Grad-CAM visualization
* Structured report
* Processing information
* Research-use disclaimer

---

## 🗂️ Study History

MongoDB stores previously analyzed studies for later retrieval.

The History module supports:

* Study search
* Finding-based filtering
* Previous result retrieval
* Record deletion
* Duplicate detection

Duplicate detection prevents the same study from being unnecessarily stored multiple times.

---

## 📊 Performance Monitoring

The prototype includes a performance module for monitoring inference processing.

During evaluation, the complete pipeline achieved an average processing time of approximately:

### **16.05 seconds per study**

This includes the main processing stages required to transform the uploaded chest X-ray into the final AI-assisted output.

The evaluation focused on whether the system could generate the complete output within a practical prototype timeframe rather than claiming real-time clinical performance.

---

## 🔁 Output Consistency

The same chest X-ray image was processed repeatedly to determine whether the system produced repeatable results.

Across five repeated runs:

* The primary prediction remained the same.
* The confidence score remained the same.
* The generated report maintained the same interpretation.
* Only small differences in processing time occurred.

This demonstrates that the inference workflow can provide **consistent and repeatable outputs for identical inputs**.

---

## ✅ Functional Testing

The prototype was tested to verify that its major functions operated as intended.

The evaluated functions included:

* Image upload
* Image display
* AI prediction
* Report generation
* PDF export
* Database storage
* Study history retrieval
* Study deletion
* API connection

All major functional test cases passed during prototype evaluation.

A successful functionality test means that the implemented system features operate according to their expected behaviour. It does **not** mean that the AI has been clinically validated.

---

## 🔗 Workflow Integration

Workflow integration evaluates whether the individual components operate together as one complete PACS-style process.

The tested workflow was:

```text
Upload
   ↓
Preprocessing
   ↓
BiomedCLIP
   ↓
Prediction
   ↓
Grad-CAM
   ↓
Qwen2.5 Report
   ↓
MongoDB Storage
   ↓
History Retrieval
   ↓
PDF Export
```

All tested workflow stages were successfully completed.

This demonstrates that the system modules can communicate and complete the end-to-end workflow without interruption during prototype testing.

---

## ⚠️ Error Handling

The system includes controlled error handling for situations such as:

* Unsupported file formats
* Invalid images
* Missing input
* Processing failures
* API communication errors

Instead of crashing, the application provides an appropriate error message to the user.

---

## 🗃️ Dataset

The project uses the **NIH ChestX-ray14** public chest X-ray dataset.

The complete dataset contains approximately **112,000 chest X-ray images** and is considerably larger than required for this prototype.

Because the project focuses on:

* System integration
* AI inference
* PACS-style workflow
* Technical functionality
* Response time
* Output consistency

rather than model training or large-scale clinical benchmarking, a **representative subset** was selected for prototype demonstration and evaluation.

---

## 🧹 Image Preprocessing

Before inference, uploaded images are prepared for BiomedCLIP.

```text
Input Chest X-Ray
        ↓
Format Validation
        ↓
RGB Preparation
        ↓
Resize to 224 × 224
        ↓
Normalization
        ↓
BiomedCLIP Inference
```

The models remain pretrained and are used for **inference only**.

---

## 🛠️ Technology Stack

| Component             | Technology                |
| --------------------- | ------------------------- |
| Programming Language  | Python                    |
| Frontend              | Streamlit                 |
| Backend API           | FastAPI                   |
| Vision-Language Model | BiomedCLIP                |
| Large Language Model  | Qwen2.5-1.5B-Instruct     |
| Model Framework       | Hugging Face Transformers |
| Database              | MongoDB                   |
| Image Storage         | GridFS                    |
| DICOM Processing      | pydicom                   |
| Explainability        | Grad-CAM                  |
| Secure Tunneling      | ngrok                     |
| Report Export         | PDF                       |
| Dataset               | NIH ChestX-ray14          |

---

## 🧪 Research Methodology

The project followed eight major phases:

```text
1. Preliminary Study
        ↓
2. Data Understanding
        ↓
3. Data Preprocessing
        ↓
4. System Design
        ↓
5. AI Pipeline Development
        ↓
6. System Implementation
        ↓
7. Prototype Testing & Evaluation
        ↓
8. Documentation
```

### 1. Preliminary Study

Reviewed chest X-ray analysis, VLMs, LLMs, explainable AI, and PACS integration.

### 2. Data Understanding

Selected and examined the NIH ChestX-ray14 dataset.

### 3. Data Preprocessing

Validated image formats, resized images to 224 × 224, and applied normalization.

### 4. System Design

Designed the system architecture, IPO framework, and PACS-style workflow.

### 5. AI Pipeline Development

Integrated BiomedCLIP and Qwen2.5 and developed the structured prompting approach.

### 6. System Implementation

Implemented the Streamlit frontend, FastAPI backend, MongoDB storage, Grad-CAM visualization, and reporting functions.

### 7. Prototype Testing and Evaluation

Evaluated functionality, error handling, response time, output consistency, stability, and workflow integration.

### 8. Documentation

Documented the implementation, findings, limitations, conclusions, and future improvements.

---

## 📈 Key Evaluation Results

| Evaluation            | Result                                        |
| --------------------- | --------------------------------------------- |
| Functional Testing    | Major test cases passed                       |
| Workflow Integration  | Complete workflow passed                      |
| Average Response Time | **16.05 seconds**                             |
| Output Consistency    | Same input produced repeatable results        |
| Error Handling        | Invalid inputs handled without system crash   |
| Stability             | Prototype completed repeated processing tests |

These results evaluate the **technical behaviour of the prototype**, not clinical diagnostic accuracy.

---

## 🌍 Research Significance

### SDG 3 — Good Health and Well-Being

The project explores how AI-assisted image interpretation and structured reporting could support more efficient and explainable medical imaging workflows.

### SDG 4 — Quality Education

The prototype can also demonstrate how AI interprets chest X-rays through predictions, confidence information, explainability visualizations, and generated reports, providing potential educational value for medical imaging and AI learning.

---

## ⚠️ Limitations

This project is a research prototype and has several important limitations:

* Uses a representative subset of a public chest X-ray dataset.
* Uses pretrained foundation models without domain-specific fine-tuning.
* Does not provide clinical diagnostic validation.
* Does not evaluate clinical metrics such as sensitivity and specificity.
* Uses a simplified PACS-style environment rather than a full hospital PACS.
* Does not provide full RIS/HL7 hospital integration.
* Generated reports are observational and non-diagnostic.
* Public datasets may not represent the full diversity of real clinical populations and imaging conditions.

---

## 🚀 Future Work

Future improvements may include:

* Fine-tuning BiomedCLIP on domain-specific chest X-ray data.
* Fine-tuning or adapting the LLM for radiology reporting.
* Testing with larger and more diverse datasets.
* Evaluating diagnostic metrics such as sensitivity and specificity.
* Validation with certified radiologists.
* Integration with real PACS environments.
* Support for hospital communication standards such as RIS/HL7.
* Improving inference speed.
* Extending the system to additional medical imaging modalities.

---

## 📁 Suggested Repository Structure

```text
project/
│
├── frontend/
│   └── streamlit_app.py
│
├── backend/
│   ├── main.py
│   ├── api/
│   └── services/
│
├── models/
│   ├── biomedclip/
│   └── qwen/
│
├── preprocessing/
│   └── image_processing.py
│
├── explainability/
│   └── gradcam.py
│
├── database/
│   └── mongodb.py
│
├── reports/
│   └── pdf_generator.py
│
├── utils/
│
├── requirements.txt
├── .gitignore
└── README.md
```

> The actual repository structure may differ depending on the implementation.

---

## 🔐 Security

Never commit private credentials to GitHub.

The following should be stored in environment variables or excluded through `.gitignore`:

```text
.env
NGROK_AUTHTOKEN
MONGODB_URI
API_KEYS
PRIVATE_CONFIG
```

If a credential has accidentally been committed, revoke or rotate it immediately rather than simply deleting it from the latest commit.

---

## 🎓 Project Information

**Final Year Project**

**Project Title:**
Chest X-Ray Diagnosis Using Vision-Language Models (VLMs) Integrated with a Picture Archiving and Communication System (PACS)

**Supervisor:**
Madam Farah Syazwani binti Mohamed Rashid

**Institution:**
Universiti Teknologi MARA (UiTM)

---

## ⚕️ Medical Disclaimer

**This software is an academic research prototype developed for educational and demonstration purposes only.**

It is **not a certified medical device**, has **not undergone formal clinical validation**, and must **not be used for diagnosis, treatment decisions, or independent clinical interpretation**.

All AI-generated predictions, heatmaps, and reports require review and confirmation by appropriately qualified healthcare professionals.

---

## 📜 License

This repository is intended primarily for academic and research purposes.

Please review the licenses and usage restrictions of all external datasets, pretrained models, libraries, and dependencies before redistributing or deploying the project.
