# AI Food Recognizer  
## 画像認識技術を活用した冷蔵庫内食材自動判別システムの開発  

- 🏆 情報処理学会 第86回大会 学生奨励賞 受賞作品  
- Final project for **"情報工学工房 (Information Engineering Workshop)"**, The University of Electro-Communications (電気通信大学)  

---

## 📹 Demo  
[![Watch the demo](https://img.youtube.com/vi/WM_rVHsI6sQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=WM_rVHsI6sQ)  

---

## 🧑‍💼 Project Info  
- **Theme:** 「巨大モデルを使いこなせ！大規模深層学習モデルを活用した画像認識・生成」  
- **Authors:**  
  - Mengchi Wang (王 孟琪)  
  - Junwen Chen (陳 俊文) – Teaching Assistant  
  - Prof. Keiji Yanai (柳井 啓司) – Faculty Advisor  

---

## 🔍 Overview  
This project integrates multiple state-of-the-art vision and language models to detect, segment, and recognize food items in images, and optionally fetch their names and calorie information using the OpenAI API.

Models used:  
- **GroundingDINO** – open-vocabulary object detection with text prompts  
- **Segment Anything (SAM)** – precise image segmentation  
- **CLIP** – feature embedding and similarity matching  
- **OpenAI GPT-4 Vision** (optional) – retrieve food names & calories  

Ideal for:  
- Food logging  
- AR food labeling  
- Diet tracking systems  

---

## 🧠 Architecture  

     +----------------+          +---------------------+
     |   Input Image  |  --->    |  GroundingDINO      |
     +----------------+          +---------------------+
                                          |
                                          v
                            +---------------------------+
                            | Segment Anything (SAM)    |
                            +---------------------------+
                                          |
                                          v
                            +---------------------------+
                            | CLIP Feature Extraction   |
                            +---------------------------+
                                          |
                  +--------------------+  v
                  | Feature Matching   | ---> Object Label
                  +--------------------+
                                          |
                                          v
                        (Optional) Call OpenAI GPT-4 API
                        to retrieve object name & calorie

---

## 📌 Features  
- Text-based object detection (GroundingDINO)  
- Image segmentation (SAM)  
- Embedding & similarity search (CLIP)  
- One-shot recognition with registered features  
- Optional: API-based food description & calories  
- Export visual results with bounding boxes & labels  

---

## 🚀 Getting Started  

### 1. Installation  
```bash
git clone https://github.com/wanaincode/KOBO2023_project.git
cd KOBO2023_project
pip install -r requirements.txt
```
Download checkpoints for:  
- GroundingDINO  
- SAM (ViT-H)  
- CLIP (ViT-B/32)  

---

### 2. Usage  

Choose one of the following modes depending on your preference and use case:

| Script                  | Interaction Type     | Detection & Recognition                | Output                              | Notes                                    |
|-------------------------|----------------------|--------------------------------------|-----------------------------------|------------------------------------------|
| `demo_ui.py` (recommended) | OpenCV GUI with buttons | Real-time detection with GUI controls | Visual output with bounding boxes and labels displayed in window; supports saving results | User-friendly, interactive, supports parameter tuning via GUI |
| `demo_no_ui.py`          | Terminal-based       | Batch or single image recognition    | Text output in terminal and saved images with annotations | Lightweight, no GUI dependencies          |
| `demo_basic.py`          | Minimal script       | Basic detection and recognition      | Saves annotated images             | Simplified, for quick testing or integration |

---

### 3. Running Recognition with `demo_ui.py` (GUI Version)  

This is the recommended way to run the recognition pipeline interactively with visual feedback.

**Steps:**  
1. Place your test images in the `./test_img/` directory.  
2. Run the GUI demo:  
   ```bash
   python src/demo_ui.py
   ```  
3. The GUI window will open, showing controls and buttons for:  
   - Loading images  
   - Adjusting detection thresholds (box and text)  
   - Running detection and segmentation  
   - Viewing recognized food items with bounding boxes and labels  
   - Saving results to `./recognized_results/`  

**Input:**  
- Images from the `test_img/` folder or loaded via GUI.  

**Output:**  
- Visual display of detected food items with segmentation masks and labels.  
- Option to save annotated images and recognition results.  

**Controls:**  
- Buttons for image navigation and detection execution.  
- Sliders or input fields to adjust detection thresholds dynamically.  

This interactive interface facilitates easy experimentation and visualization without requiring command-line parameter tuning.

---

## 📄 License  

This project is released under the MIT License.  

---
