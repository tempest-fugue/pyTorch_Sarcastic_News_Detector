# 🤖 Sarcastic News Headline Detector
This project uses natural language processing (NLP) and deep learning to detect sarcasm in news headlines. It's a binary classification task aimed at identifying whether a given headline is sarcastic or not, using a dataset curated for this specific problem.

# 📌 Project Overview
The goal is to build and train a neural network model capable of understanding the subtle cues that distinguish sarcasm from literal text. It involves:

Text preprocessing and cleaning

Tokenization and padding

Building a PyTorch-based LSTM model

Training and evaluating the model on labeled data

Saving the best-performing model for inference

# 📂 Project Structure
```
├── sarcastic_news_headline_detector.ipynb  # Main notebook with data, modeling, and evaluation
├── README.md                               # Project overview and instructions
├── LICENSE                                 # License info (MIT by default)
├── /src
│   └── my_dataset.py                       # Custom dataset class and utilities
🧠 Model Architecture
```
Embedding Layer: Converts words into dense vectors

LSTM Layer: Captures temporal dependencies in the sequence

Fully Connected Layer: Outputs logits for binary classification

Loss Function: Binary Cross-Entropy Loss

Optimizer: Adam

# 📊 Dataset
The dataset includes sarcastic and non-sarcastic news headlines, each labeled appropriately. It is split into training, validation, and test sets with preprocessing handled in the notebook and my_dataset.py.

# 🚀 How to Run
Clone the repository:
```
git clone https://github.com/your-username/pyTorch_Sarcastic_News_Detector.git
cd pyTorch_Sarcastic_News_Detector
```
Install dependencies:

```
pip install -r requirements.txt
```

Run the notebook:
Open sarcastic_news_headline_detector.ipynb in Jupyter or VS Code and run cells sequentially.

# 🧪 Evaluation Metrics
Accuracy

Precision / Recall / F1-score

Confusion Matrix

ROC-AUC Curve

# 💾 Model Saving
The best model (based on validation F1-score) is saved as best_model.pth and can be loaded for final evaluation or inference.

# 🔧 Requirements
Python 3.8+

PyTorch

scikit-learn

pandas

numpy

rich

(Add requirements.txt if not already created.)

# 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

# 🤝 Contributing
Pull requests and collaborations are welcome! Please open an issue to discuss your proposed changes before submitting a PR.

Let me know if you'd like a version with badges (e.g. PyTorch, license, etc.) or additional setup instructions for the src/ folder.
# PyTorch Sarcastic News Headline Detector
Notebook uses a custom pyTorch training/validation loop modifying a foundational LLM (BERT) to detect sarcastic news headlines. This represents a NLP classification task.

### Model Train/Val Performance 

|   Epoch |   Train Loss |   Loss Val |   Train Accuracy |   Val Accuracy |   Train F1 |   Val F1 |   Train AUC_ROC |   Val AUC_ROC |   Training Time |
|---------|--------------|------------|------------------|----------------|------------|----------|-----------------|---------------|-----------------|
|       1 |       0.4783 |     0.4068 |         0.78683  |       0.816678 |     0.7864 |   0.816  |          0.8627 |        0.8995 |         9.82776 |
|       2 |       0.4125 |     0.3719 |         0.822525 |       0.836478 |     0.8224 |   0.8365 |          0.8952 |        0.9147 |         9.77665 |
|       3 |       0.3928 |     0.365  |         0.829364 |       0.833217 |     0.8293 |   0.8323 |          0.9048 |        0.9216 |        18.292   |
|       4 |       0.3855 |     0.3573 |         0.833458 |       0.837876 |     0.8334 |   0.8367 |          0.9082 |        0.9273 |        12.0928  |
|       5 |       0.3853 |     0.3431 |         0.831711 |       0.844631 |     0.8317 |   0.8442 |          0.9081 |        0.929  |        13.9551  |
|       **6** |       **0.3802** |     **0.3425** |         **0.833358** |       **0.852551** |     **0.8333** |   **0.8526** |          **0.9107** |        **0.93**   |        **11.6359**  |
|       7 |       0.3773 |     0.3528 |         0.835156 |       0.839273 |     0.8351 |   0.8379 |          0.9118 |        0.9313 |        30.4976  |
|       8 |       0.3769 |     0.3358 |         0.837901 |       0.848824 |     0.8379 |   0.8485 |          0.9125 |        0.932  |        17.7042  |
|       9 |       0.3746 |     0.3336 |         0.836453 |       0.852551 |     0.8364 |   0.8526 |          0.9134 |        0.9317 |        18.8004  |
|      10 |       0.3735 |     0.3359 |         0.837552 |       0.844864 |     0.8375 |   0.8445 |          0.9138 |        0.932  |        17.2321  |


📦 Dataset Access
This project uses a dataset hosted on Kaggle and accessed programmatically using the opendatasets library. The dataset is not included in this repository to reduce size and avoid licensing issues.

To download the dataset automatically when running the notebook, you’ll need to set up Kaggle API access.

🔑 Kaggle API Setup
Create a Kaggle account (if you don’t have one): https://www.kaggle.com/account

Generate your API token:

Go to your Kaggle Account settings

Scroll down to the "API" section

Click "Create New API Token"

A file called kaggle.json will be downloaded

Place kaggle.json in your environment:

For Jupyter or Colab notebooks, upload the kaggle.json file, then run:

```import os
os.environ['KAGGLE_CONFIG_DIR'] = '/path/to/your/json/' ```
Alternatively, place kaggle.json in your home directory under ~/.kaggle/

Install and use opendatasets to download the dataset:

```!pip install opendatasets
import opendatasets as od
od.download("https://www.kaggle.com/dataset-url")```
⚠️ Note: If you're running this in an environment like Google Colab, remember to re-upload kaggle.json each session unless you're mounting from Google Drive.

Let me know if you'd like this tailored for Colab-only, or with the actual dataset name/URL filled in.







