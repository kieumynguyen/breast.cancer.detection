 # Breast Cancer Detection
# **Introduction**
<img src="https://imgur.com/OupRcxj"
# **Installation**
Create virtual environment
To set up your python environment to run the code in this repository, follow the instructions below.
1. *Create (and activate) a new environment with Python 3.6.*
Linux or Mac:
```bash
conda create --name bcdp python=3.8
conda activate bcdp
```   
Windows:
```bash
conda create --name bcdp python=3.8
conda activate bcdp
```
2. *Clone the repository (if you haven't already!), and navigate to the breast.cancer.detection folder. Then, install several dependencies.*
```bash
git clone https://github.com/kieumynguyen/breast.cancer.detection.git
cd breast-cancer-detection
pip install -r requirements.txt
```

# **Download and information of dataset**

Go to folder data, there are 4 files (xtest, ytest, xtrain, ytrain).

The Breast Cancer Wisconsin Dataset have 569 samples, 31 index.

Sorting it into training and test sets with the 'input' values to the Neural Network as 'X' values

The expected 'output' (a 0 if benign and a 1 if malignant) as the 'Y' values.

There are 2 classes, 4 layers.

# **Usage**

# **How to run**
```bash
python train.py
```

# **Expected output:**


*Make sure:*
the dataset follow the structure described below
folders models (for training) and results (testing) exists
After the models are trained, run:

```bash
python test.py
```
