# Cancer_Detection
Breast cancer detection using LLM
# Cancer Detection Using Breast Ultrasound Images

## Project Overview

This repository contains a breast cancer detection pipeline that leverages breast ultrasound images and applies machine learning techniques to classify images into three categories:

* **Normal**
* **Benign**
* **Malignant**

Early and accurate detection of breast cancer can significantly improve patient outcomes. This project demonstrates an end-to-end workflow — from data preprocessing and exploratory analysis to model training, evaluation, and visualization.

## Repository Structure

```
├── data/
│   └── breast_ultrasound_images/        # Raw PNG images organized by class
├── notebooks/
│   └── Cancer_Detection.ipynb           # Jupyter notebook with full analysis and modeling
├── src/
│   ├── data_preprocessing.py            # Scripts for loading and preprocessing images
│   ├── model.py                         # Model definitions and training routines
│   └── utils.py                         # Utility functions (metrics, plotting, etc.)
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation (this file)
```

## Dataset

* **Source:** Al‑Dhabyani W., Gomaa M., Khaled H., Fahmy A. (2020). Dataset of Breast Ultrasound Images. *Data in Brief*, 28:104863.
* **Description:** 780 grayscale ultrasound images (PNG) from 600 female patients aged 25–75. Images are labeled as normal, benign, or malignant.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/breast-cancer-detection.git
   cd breast-cancer-detection
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\\Scripts\\activate     # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the data:**

   * Place the downloaded dataset under `data/breast_ultrasound_images/`, preserving the `normal/`, `benign/`, and `malignant/` subfolders.

2. **Run the notebook:**

   ```bash
   jupyter notebook notebooks/Cancer_Detection.ipynb
   ```

   The notebook sequentially:

   * Loads and explores the image data
   * Performs preprocessing (resizing, normalization)
   * Defines and trains a convolutional neural network (CNN)
   * Evaluates model performance
   * Visualizes results (confusion matrix, sample predictions)

3. **Scripts (optional):**
   To run preprocessing or model training as standalone scripts:

   ```bash
   python src/data_preprocessing.py   # Preprocess and save cleaned data
   python src/model.py                # Train and evaluate the model
   ```

## Results

* **Model Accuracy:** Approximately 78% on the test split.
* **Confusion Matrix:** Available in the notebook under **Model Evaluation**.
* **Sample Predictions:** Visualized for qualitative assessment.

## Future Work

* Experiment with advanced architectures (e.g., transfer learning with ResNet or EfficientNet).
* Implement data augmentation to address class imbalance.
* Optimize hyperparameters using grid search or Bayesian optimization.
* Deploy model inference as a REST API.

## Citation

If you use this work, please cite the original dataset and this repository:

> Al‑Dhabyani W., Gomaa M., Khaled H., Fahmy A. Dataset of Breast Ultrasound Images. *Data in Brief*. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
