# 画像認識技術を活用した冷蔵庫内食材自動判別システムの開発   
## 1. Intro

- 🏆 情報処理学会 第86回大会 学生奨励賞 受賞作品  
- Final project for **"情報工学工房 (Information Engineering Workshop)"**, The University of Electro-Communications (電気通信大学)  


### 1.1 📹 Demo  
[![Watch the demo](https://img.youtube.com/vi/WM_rVHsI6sQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=WM_rVHsI6sQ)  

---

### 1.2 🧑‍💼 Project Info  
- **Theme:** 「巨大モデルを使いこなせ！大規模深層学習モデルを活用した画像認識・生成」  
- **Authors:**  
  - Mengchi Wang (王 孟琪)  
  - Junwen Chen (陳 俊文) – Teaching Assistant  
  - Prof. Keiji Yanai (柳井 啓司) – Faculty Advisor  

---

### 1.3 🔍 Overview  
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

## 2. Architecture  

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



### 2.1 🔍 Key Features
- Text-based object detection (GroundingDINO)  
- Image segmentation (SAM)  
- Embedding & similarity search (CLIP)  
- One-shot recognition with registered features  
- Optional: API-based food description & calories  
- Export visual results with bounding boxes & labels  

---

## 3. Getting Started  

### 3.1 Installation  
```bash
git clone https://github.com/wanaincode/KOBO2023_project.git
cd KOBO2023_project
pip install -r requirements.txt
```
Download checkpoints for:  
- GroundingDINO  
- SAM (ViT-H)  
- CLIP (ViT-B/32)  

All scripts expect read/write under `data/` (created automatically on first run).


### 3.2 Usage  

Choose one of the following modes depending on your preference and use case:

| Script                     | Interaction Type       | Detection & Recognition                      | Output & Paths (under `data/`)                                  | Notes |
|---------------------------|------------------------|-----------------------------------------------|------------------------------------------------------------------|-------|
| `demo_ui.py` (recommended)| OpenCV GUI (webcam)    | Real-time detect → segment → embed → match    | Saves to `recognized_results/`; uses `sample_features/` or `extract_saved_obj_feature/` | Mouse buttons: Abstract / Recognize; Enter to capture |
| `demo_no_ui.py`           | Terminal + webcam      | Batch/one-shot via key commands               | Same `data/` layout (`saved_obj_img/`, `extract_saved_obj_feature/`, etc.) | Lightweight, no GUI |
| `demo_basic.py`           | Minimal (webcam)       | Basic CLIP similarity example                 | Writes simple outputs under `data/`                              | For quick testing |


#### 📂 Directory Structure

```
KOBO2023_project/
├── src/                       # Source code
│   ├── api_get_food_info.py
│   ├── demo_ui.py
│   ├── demo_no_ui.py
│   ├── opencv_frame.py
│   └── sam_generate_obj_mask.py
├── notebooks/                 # Jupyter notebooks
│   ├── SAM_demo.ipynb
│   ├── clip_test.ipynb
│   └── ground_dino_sam_clip.ipynb
├── data/
│   ├── saved_obj_img/         # Your captured object images (organized by object name)
│   ├── extract_saved_obj_feature/  # Auto-generated masks/crops & obj_features.pt
│   ├── sample_features/       # Bundled sample obj_features.pt (for quick try)
│   ├── test_img/              # Auto-saved test.jpg when recognizing
│   └── recognized_results/    # Prediction results (annotated images)
├── requirements.txt
└── README.md
```

> Note: If you are running scripts directly from the repo root (not using `src/`), adjust the command accordingly (e.g., `python demo_ui.py`). All I/O paths are under the `data/` directory.

### 3.3 Running Recognition with `demo_ui.py` (GUI Version)

This is the recommended way to run the pipeline interactively with visual feedback. The GUI uses your webcam.

#### 💡 Quick Try  
You can try recognition immediately using the bundled sample features:  
- The app will prefer `data/sample_features/obj_features.pt` if present;  
- Otherwise, it will fall back to `data/extract_saved_obj_feature/obj_features.pt`.  

A snapshot from the webcam will be saved under `data/test_img/`, and annotated results will be written to `data/recognized_results/`.

#### 💡 Register Your Own Objects  
1. Run the GUI:
   ```bash
   python src/demo_ui.py
   ```  
2. In the bottom input box, type an **object name** (e.g., `apple`) and press **Enter**. The program will capture **3 directions** (`x`, `y`, `z`) and save images under `data/saved_obj_img/<object_name>/`.  
3. Repeat step 2 for more objects.  
4. Click **Abstract** (right panel) to extract features, which will create `data/extract_saved_obj_feature/obj_features.pt`.  
5. Click **Recognize** to run detection/segmentation and show similarity scores per detected region. Results are saved to `data/recognized_results/`.

#### 💡 Controls  
- **Enter**: confirm the typed object name and start capturing images  
- **Buttons**: `Abstract` (feature extraction), `Recognize` (run recognition), `Exit` (quit)  
- **ESC**: quit (when idle/Waiting)  

#### 💡 Outputs (summary)  
- Training images: `data/saved_obj_img/`  
- Extracted features: `data/extract_saved_obj_feature/`  
- Sample features: `data/sample_features/`  
- Test snapshot: `data/test_img/`  
- Recognition results: `data/recognized_results/`  

---
