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

'''import os
os.environ['KAGGLE_CONFIG_DIR'] = '/path/to/your/json/'
Alternatively, place kaggle.json in your home directory under ~/.kaggle/'''

Install and use opendatasets to download the dataset:

'''!pip install opendatasets
import opendatasets as od
od.download("https://www.kaggle.com/dataset-url")'''
⚠️ Note: If you're running this in an environment like Google Colab, remember to re-upload kaggle.json each session unless you're mounting from Google Drive.

Let me know if you'd like this tailored for Colab-only, or with the actual dataset name/URL filled in.







