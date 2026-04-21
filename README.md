# Disease Prediction System

An AI-powered disease prediction web app using Flask + Machine Learning.

## Quick Start

### Windows
1. Double-click `START_WINDOWS.bat`
2. Open browser → http://127.0.0.1:5000

### Mac / Linux
1. Open terminal in this folder
2. Run: `bash START_MAC_LINUX.sh`
3. Open browser → http://127.0.0.1:5000

## Manual Setup (All Platforms)

```
# 1. Create virtual environment
python -m venv venv

# 2. Activate (Windows)
venv\Scripts\activate.bat

# 2. Activate (Mac/Linux)
source venv/bin/activate

# 3. Install libraries
pip install -r requirements.txt

# 4. Train the model (run ONCE)
python backend/train_model.py

# 5. Start the app
python backend/app.py

# 6. Open browser
http://127.0.0.1:5000
```

## Project Structure

```
disease-prediction/
├── backend/
│   ├── app.py            ← Flask web server
│   └── train_model.py    ← ML model training
├── templates/
│   └── index.html        ← Main frontend page
├── static/
│   ├── style.css         ← Dark theme styles
│   └── script.js         ← Frontend logic + charts
├── models/               ← Auto-created after training
├── dataset/              ← Auto-created after training
├── requirements.txt      ← Python dependencies
├── START_WINDOWS.bat     ← One-click Windows launcher
└── START_MAC_LINUX.sh    ← One-click Mac/Linux launcher
```

## Features
- 20 diseases, 45 symptoms
- 5 ML models compared (Random Forest, SVM, KNN, etc.)
- Dark medical dashboard UI
- Confidence score + medical advice
- Interactive charts (Chart.js)
