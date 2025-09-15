# 画像認識技術を活用した冷蔵庫内食材自動判別システムの開発 (日本語版)   
## 1. はじめに

- 🏆 情報処理学会 第86回大会 学生奨励賞 受賞作品  
- 電気通信大学「情報工学工房」最終プロジェクト
- 調布祭2023に展示    

### 1.1 デモ  
[![デモを見る](https://img.youtube.com/vi/WM_rVHsI6sQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=WM_rVHsI6sQ)  


### 1.2 プロジェクト情報  
- **テーマ:** 「巨大モデルを使いこなせ！大規模深層学習モデルを活用した画像認識・生成」  
- **著者:**  
  - 王 孟琪 (Mengchi Wang)  
  - 陳 俊文 (Junwen Chen) – TAさん  
  - 柳井 啓司 (Prof. Keiji Yanai) – 指導教授  


### 1.3 概要  
本プロジェクトは、最先端の視覚と言語モデルを統合し、画像内の食材を検出・セグメント化・認識し、必要に応じてOpenAI APIで名称やカロリー情報を取得する。

使用モデル:  
- **GroundingDINO** – テキストプロンプトによるオープンボキャブラリ物体検出  
- **Segment Anything (SAM)** – 高精度画像セグメンテーション  
- **CLIP** – 特徴抽出と類似度マッチング  
- **OpenAI GPT-4 Vision**（オプション） – 食材名とカロリーの取得  

用途例:  
- 食事記録  
- AR食材ラベリング  
- ダイエット管理システム  

---

## 2. アーキテクチャ  

     +----------------+          +---------------------+
     |   入力画像     |  --->    |  GroundingDINO      |
     +----------------+          +---------------------+
                                          |
                                          v
                            +---------------------------+
                            | Segment Anything (SAM)    |
                            +---------------------------+
                                          |
                                          v
                            +---------------------------+
                            | CLIP特徴抽出              |
                            +---------------------------+
                                          |
                  +--------------------+  v
                  | 特徴マッチング      | ---> 物体ラベル
                  +--------------------+
                                          |
                                          v
                        （オプション）OpenAI GPT-4 APIを呼び出し
                        物体名・カロリーを取得



### 2.1 特徴
- テキストベース物体検出（GroundingDINO）  
- 画像セグメンテーション（SAM）  
- 埋め込み特徴と類似度検索（CLIP）  
- ワンショット認識（登録済み特徴を使用）  
- オプション：APIによる食材説明とカロリー情報  
- バウンディングボックスとラベル付きの可視化結果出力  

---

## 3. 使い方  

### 3.1 インストール  
```bash
git clone https://github.com/wanaincode/KOBO2023_project.git
cd KOBO2023_project
pip install -r requirements.txt
```
以下のチェックポイントをダウンロードしてください:  
- GroundingDINO  
- SAM (ViT-H)  
- CLIP (ViT-B/32)  

すべてのスクリプトは `data/` フォルダ内の読み書きを想定しており、初回実行時に自動生成される。


### 3.2 利用方法  

利用目的に応じて以下のモードから選択してください:

| スクリプト                 | インタラクションタイプ | 検出・認識内容                             | 出力・パス (`data/` 以下)                                  | 備考 |
|---------------------------|------------------------|------------------------------------------|------------------------------------------------------------|-------|
| `demo_ui.py` (推奨)        | OpenCV GUI (ウェブカメラ) | リアルタイム検出→セグメント→埋め込み→マッチング | `recognized_results/` に保存。`sample_features/` または `extract_saved_obj_feature/` 使用 | マウスボタン: 抽象化 / 認識; Enterでキャプチャ |
| `demo_no_ui.py`            | ターミナル + ウェブカメラ | キー操作によるバッチ/ワンショット認識 | 同じく `data/` 配下（`saved_obj_img/`, `extract_saved_obj_feature/` 等） | 軽量、GUIなし |
| `demo_basic.py`            | 最小限 (ウェブカメラ)     | CLIP類似度の基本例                      | `data/` 以下に簡易出力                                     | 簡易テスト用 |


#### ディレクトリ構成

```
KOBO2023_project/
├── src/                       # ソースコード
│   ├── api_get_food_info.py
│   ├── demo_ui.py
│   ├── demo_no_ui.py
│   ├── opencv_frame.py
│   └── sam_generate_obj_mask.py
├── notebooks/                 # Jupyterノートブック
│   ├── SAM_demo.ipynb
│   ├── clip_test.ipynb
│   └── ground_dino_sam_clip.ipynb
├── data/
│   ├── saved_obj_img/         # キャプチャ済みオブジェクト画像（名前別フォルダ）
│   ├── extract_saved_obj_feature/  # 自動生成されたマスク/切り出し & obj_features.pt
│   ├── sample_features/       # バンドル済みサンプルobj_features.pt（試用用）
│   ├── test_img/              # 認識時に自動保存されるtest.jpg
│   └── recognized_results/    # 予測結果（注釈付き画像）
├── requirements.txt
└── README.md
```

> 注意: リポジトリ直下から直接スクリプトを実行する場合は（`src/`を経由しない場合）、コマンドを適宜調整してください（例: `python demo_ui.py`）。すべての入出力は `data/` ディレクトリ内にある。

### 3.3 GUI版による認識の実行 (`demo_ui.py`)

インタラクティブに視覚フィードバックを得ながら実行する推奨方法。ウェブカメラを使用する。

#### 💡 すぐに試す  
バンドル済みのサンプル特徴量を使ってすぐに認識を試せます:  
- まず `data/sample_features/obj_features.pt` を優先使用;  
- なければ `data/extract_saved_obj_feature/obj_features.pt` を使用。  

ウェブカメラからのスナップショットは `data/test_img/` に保存され、注釈付き結果は `data/recognized_results/` に書き込まれる。

#### 💡 自分のオブジェクトを登録する  
1. GUIを起動:  
   ```bash
   python src/demo_ui.py
   ```  
2. 下部の入力欄にオブジェクト名（例: `apple`）を入力し、Enterキーを押す。  
   3方向（`x`, `y`, `z`）の画像をキャプチャし、`data/saved_obj_img/<object_name>/` に保存。  
3. 他のオブジェクトも同様に登録。  
4. 右パネルの「抽象化」ボタンを押して特徴を抽出し、`data/extract_saved_obj_feature/obj_features.pt` を生成。  
5. 「認識」ボタンで検出・セグメント・類似度スコア表示を実行。結果は `data/recognized_results/` に保存。

#### 💡 操作方法  
- **Enter**: 入力したオブジェクト名を確定し、画像キャプチャ開始  
- **ボタン**: `抽象化`（特徴抽出）、`認識`（認識実行）、`終了`（終了）  
- **ESC**: 待機中に終了  

#### 💡 出力  
- 学習用画像: `data/saved_obj_img/`  
- 抽出特徴: `data/extract_saved_obj_feature/`  
- サンプル特徴: `data/sample_features/`  
- テストスナップショット: `data/test_img/`  
- 認識結果: `data/recognized_results/`  

---

# English Version

# AI Food Recognizer: Refrigerator Ingredient Recognition System  
## 1. Intro

- 🏆 Student Encouragement Award at the 86th Annual Conference of the Information Processing Society of Japan (IPSJ)
- Final project for **"情報工学工房 (Information Engineering Workshop)"**, The University of Electro-Communications (電気通信大学)  
- Exhibited at Chofu Festival 2023


### 1.1 Demo  
[![Watch the demo](https://img.youtube.com/vi/WM_rVHsI6sQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=WM_rVHsI6sQ)  


### 1.2 Project Info  
- **Theme:** 「巨大モデルを使いこなせ！大規模深層学習モデルを活用した画像認識・生成」  
- **Authors:**  
  - Mengchi Wang (王 孟琪)  
  - Junwen Chen (陳 俊文) – Teaching Assistant  
  - Prof. Keiji Yanai (柳井 啓司) – Faculty Advisor  


### 1.3 Overview  
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



### 2.1 Key Features
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


#### Directory Structure

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
