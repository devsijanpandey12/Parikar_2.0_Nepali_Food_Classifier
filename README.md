# Parikar 2.0 (परिकार) - Nepali Culinary Vision AI 🥟🏔️

An interactive, web-deployable Nepali Food Classification Platform powered by next-generation Computer Vision models (**MobileNetV4**, **ConvNeXt-V2**, and **EfficientNetV2**).

This project upgrades the original research and mobile app into a modern, 24/7 cloud-deployable web application that anyone can visit to classify traditional Nepali cuisine using their smartphone or computer camera.

---

## 🌟 Key Features

* **3 Modern Vision Backbones (2024–2026):**
  * `MobileNetV4-Conv`: Ultra-fast sub-15ms edge inference.
  * `ConvNeXt-V2-Nano`: Pure CNN with modern Vision Transformer design principles.
  * `EfficientNetV2-S`: Deep progressive training for fine-grained food textures.
* **Dual Input Modes:**
  * **Drag & Drop / File Upload:** Fast photo input with live preview.
  * **Realtime Camera / Webcam:** In-browser video stream with mobile front/back camera flipping.
* **1-Click Quick Test Badges:** Built-in high-resolution test photos for Momo, Selroti, Dal Bhat, Gundruk, Yomari, and Sekuwa.
* **Rich Culinary Knowledge Base:**
  * Dish name in English and Nepali Devanagari script (e.g. मोमो, सेलरोटी, धिँडो).
  * Origin and cultural festive heritage.
  * Key traditional Himalayan ingredients.
  * Traditional preparation & recommended accompaniments (e.g. Timur ko achar).
* **Deployable Anywhere:**
  * Works autonomously 24/7 on cloud hosting.
  * Can be hosted for free on **GitHub Pages**, **Vercel**, **Heroku**, or **DigitalOcean**.

---

## 📂 Project Architecture

```
new_project/
├── backend/
│   ├── app.py                # FastAPI server (Endpoints: /api/predict, /api/models, /api/dishes)
│   ├── models_handler.py     # Inference manager supporting PyTorch, ONNX, and features
│   ├── dishes_data.json      # Comprehensive metadata database for 16 Nepali dishes
│   ├── requirements.txt      # Python dependencies
│   ├── Procfile              # Heroku deployment configuration
│   └── Dockerfile            # Container deployment (DigitalOcean / Render / HuggingFace)
├── frontend/
│   ├── index.html            # Modern web interface (HTML5 semantic)
│   ├── style.css             # Himalayan Night & Saffron glassmorphic design system
│   ├── app.js                # Frontend engine (Camera, drag-drop, API, encyclopedia)
│   └── assets/               # High-resolution food sample photos
├── scripts/
│   ├── dataset_collector.py  # Image harvester with Google SigLIP / CLIP zero-shot cleaner
│   └── train_models.py       # PyTorch/Timm training pipeline with ONNX auto-export
└── README.md
```

---

## 🚀 Running Locally

### Option 1: Run Full Web App + Backend Server (Recommended)
```bash
# 1. Navigate to backend directory
cd backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
python -m uvicorn app:app --reload --port 8000
```
Open your browser at: **`http://localhost:8000`**

### Option 2: Run Frontend Directly in Browser
You can open `frontend/index.html` directly in any web browser! The application includes a built-in client engine that simulates predictions and demonstrates all features offline without needing the backend server.

---

## 🎓 Hosting with GitHub Student Developer Pack

You have free student benefits that make deploying this project completely free:

### 1. Free Frontend Hosting (GitHub Pages / Vercel)
* **GitHub Pages:**
  1. Push `new_project` to your GitHub repository.
  2. In your repo settings, go to **Pages** -> Source: `Deploy from a branch` -> Folder: `/frontend`.
* **Vercel:**
  1. Connect your GitHub repository to [Vercel](https://vercel.com).
  2. Select root directory as `frontend` and click **Deploy**.

### 2. Autonomous Cloud API Backend (Heroku / DigitalOcean)
* **Heroku (Using your $13/month student credit):**
  1. Install Heroku CLI.
  2. Run:
     ```bash
     cd backend
     heroku create parikar-api
     git push heroku main
     ```
  3. The API will run **24/7 autonomously without needing your laptop**.
* **DigitalOcean (Using your $100–$200 student credit):**
  1. Create an App on DigitalOcean App Platform using the provided `Dockerfile`.

### 3. Custom Domain (.me from Namecheap)
* Claim your free `.me` domain from Namecheap via the [GitHub Student Developer Pack](https://education.github.com/pack).
* Point the CNAME record to your Vercel / GitHub Pages deployment (e.g. `parikar.me`).

---

## 🔬 Dataset & Model Retraining Guide

### Step 1: Collect & Clean Data with SigLIP
```bash
cd scripts
python dataset_collector.py
```
This automatically queries candidate images and uses **Google SigLIP / OpenAI CLIP** to reject noisy or non-food images.

### Step 2: Train & Export to ONNX
```bash
cd scripts
python train_models.py --model convnext_v2 --epochs 25
```
Exported `.onnx` model weights are automatically saved to `backend/weights/convnext_v2.onnx` for web serving!
