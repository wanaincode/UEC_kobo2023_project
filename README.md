### 画像認識技術を活用した冷蔵庫内食材自動判別システムの開発  
- 情報処理第86回大会 学生奨励賞 受賞作品
- AI Food Recognizer with GroundingDINO + SAM + CLIP

This project is the final work for the elective course **"情報工学工房 (Information Engineering Workshop)"** at **The University of Electro-Communications (電気通信大学)**, under the I-Class curriculum for Information Science students.

🧑‍💼 Theme:  
**「巨大モデルを使いこなせ！大規模深層学習モデルを活用した画像認識・生成」**  
("Master the Giant Models! Image Recognition and Generation with Large-Scale Deep Learning Models")

👨‍💻 Authors:  
- Mengchi Wang (王 孟琪)  
- Junwen Chen (陳 俊文) – Teaching Assistant  
- Prof. Keiji Yanai (柳井 啓司) – Faculty Advisor

---

This project integrates multiple state-of-the-art vision and language models to detect, segment, and recognize food items in images, and optionally fetch their names and calorie information using the OpenAI API.

---

🔍 Overview

This pipeline combines:
- **GroundingDINO** – for open-vocabulary object detection with text prompts  
- **Segment Anything (SAM)** – for precise image segmentation  
- **CLIP** – for embedding extracted image regions into a semantic space  
- **OpenAI GPT-4o API** – to describe the recognized object and return its calorie (optional)

Ideal for building intelligent food loggers, AR food labeling, and diet tracking systems.

---

🧠 Architecture

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

📌 Features
- Text-based object detection using GroundingDINO
- Image segmentation using SAM (Segment Anything)
- Feature embedding and similarity matching with CLIP
- Optional: retrieve object description and calories via OpenAI GPT-4 Vision
- Feature registration and one-shot object recognition
- Exportable visual results with bounding boxes and labels

---

🚀 Getting Started

1. Install Dependencies

Install Python dependencies and download model checkpoints for:
- GroundingDINO
- SAM (ViT-H)
- CLIP (ViT-B/32)
- OpenAI SDK (optional)

```python
pip install -r requirements.txt
```

---

2. Run as Needed

Choose the appropriate script depending on your needs:
-  ground_dino_sam_clip.ipynb:
Interactive notebook for testing the combination of GroundingDINO, SAM, and CLIP.
- 💻 demo_ver2_without UI.py:
Terminal-based script. You manually type the object name, extract features, and recognize.
- demo_ver3_UI.py (Recommended):
OpenCV GUI-based version that lets you input object names and trigger feature extraction/recognition via buttons.
- API_get_food_info.py:
Upload test_item.jpg into the folder, and this script will send it to GPT-4 Vision using your own OpenAI API key to return the food name and calorie.

📂 Directory Structure
```
.
├── GroundingDINO/                # GroundingDINO model & configs
├── segment-anything/             # SAM model
├── saved_obj_img/                # Registered training images
├── extract_saved_obj_feature/    # Feature vectors
├── test_img/                     # Input test images
├── recognized_results/           # Prediction result images
├── main.py                       # Main logic script
├── demo_ver3_UI.py               # OpenCV UI version
├── API_get_food_info.py          # OpenAI calorie recognition
└── requirements.txt
```

---

3. Run Recognition

Put test images in test_img/, and run the recognition pipeline:
```python
recognize_pipeline(model, predictor, extractor, obj_features, './test_img/example.jpg', box_threshold=0.15, text_threshold=0.15)
```
The results will be saved under recognized_results/.

---

📂 Directory Structure
```
.
├── GroundingDINO/                # GroundingDINO model & configs
├── segment-anything/             # SAM model
├── saved_obj_img/                # Registered training images
├── extract_saved_obj_feature/    # Feature vectors
├── test_img/                     # Input test images
├── recognized_results/           # Prediction result images
├── main.py                       # Main logic script
└── utils/                        # (Optional) helper functions
```

---

🔑 API Integration (Optional)

To fetch object name and calorie:

1. Set your OpenAI API key:

```python
client = OpenAI(api_key="your-api-key")
```

2. The model sends a base64-encoded cropped image to GPT-4 Vision, requesting name and calorie in a defined format.

---

📸 Example Results

[![Watch the demo](https://img.youtube.com/vi/WM_rVHsI6sQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=WM_rVHsI6sQ)

> Click the image above to watch a demo on YouTube.

---

📄 License

This project is released under the MIT License.

