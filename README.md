# Multi-Attribute-Speech-Analysis-System

Using a multi-task deep learning architecture, we aim to develop a comprehensive speech analysis system that simultaneously extracts demographic information, accent features, and text content from audio input, enabling more personalized and efficient voice-based interactions across industries.

## 🎯 Project Overview

This project implements a speech analysis system capable of predicting multiple attributes from audio:
- **Age**: Predicting speaker age from voice characteristics
- **Gender**: Identifying speaker gender based on audio features
- **Accent**: Classifying regional accent patterns
- **Text**: Transcribing spoken content from audio features

The system leverages the Mozilla Common Voice dataset and extracts various audio features including MFCCs, spectral centroids, RMS values, and mel energy measurements to train separate deep neural network models for each attribute.

## 🔑 Key Features

- **Multi-attribute analysis**: Predict age, gender, accent, and transcribed text from a single audio input
- **Customized preprocessing**: Tailored preprocessing pipelines for each attribute prediction task
- **Feature engineering**: Extraction of 13 MFCC coefficients (mean and std), spectral features, and energy metrics
- **Deep learning models**: Specialized neural network architectures for each prediction task
- **Comprehensive evaluation**: Task-specific metrics to evaluate model performance

## 📂 Repository Structure

* `data/`: Raw and processed datasets
  * `raw/`: Original CSV files from Common Voice dataset
  * `processed/`: Preprocessed data ready for model training
* `notebooks/`: Jupyter Notebooks for data validation and analysis
  * `preprocessing/`: Notebooks for data cleaning and feature extraction
  * `models/`: Notebooks for model development and evaluation
* `src/`: Preprocessing or helper scripts
* `data_card.md`: Dataset documentation (Google Data Card)
* `requirements.txt`: List of required Python packages
* `README.md`: This file

## 🚀 Installation Guide

### 🔧 Requirements

* Python 3.7 or higher
* All dependencies listed in `requirements.txt`

### 📦 Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/roshan-rs-git/cs5661-dsproject-Multi-Attribute-Speech-Analysis-System.git
cd cs5661-dsproject-Multi-Attribute-Speech-Analysis-System
```

2. **Create and activate a virtual environment**

```bash
# pip install virtualenv (if you don't have virtualenv installed)
virtualenv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install project dependencies**

```bash
pip install -r requirements.txt
```

## 📊 Dataset

This project uses the [Mozilla Common Voice dataset](https://www.kaggle.com/datasets/mozillaorg/common-voice/data) available on Kaggle. Common Voice is a corpus of speech data read by users on the Common Voice website (http://voice.mozilla.org/), based upon text from public domain sources.

The dataset is organized with the following attributes:
- **filename**: Relative path of the audio file
- **text**: Transcription of the audio
- **up_votes**: Number of people who confirmed audio matches the text
- **down_votes**: Number of people who indicated audio does not match text
- **age**: Age range of the speaker (teens, twenties, thirties, etc.)
- **gender**: Gender of the speaker (male, female, other)
- **accent**: Accent of the speaker (us, australia, england, canada, etc.)
- **duration**: Length of the audio clip

The dataset is split into several subsets:
- **Valid**: Audio clips verified by at least 2 people where the majority confirm the audio matches the text
- **Invalid**: Clips where the majority indicate the audio does not match the text
- **Other**: Clips with fewer than 2 votes or equal valid/invalid votes

Each subset is further divided into development, training, and testing groups.

### Data Preprocessing

For each attribute prediction task, we perform specialized preprocessing:

1. **Age, Gender, Accent**:
   - Extract audio features (MFCCs, spectral features, RMS)
   - Remove irrelevant columns for each task
   - Drop records with NaN values

2. **Text**:
   - Filter for high-quality samples (upvotes ≥ downvotes)
   - Extract audio features
   - Remove irrelevant columns
   - Drop records with NaN values

## 📈 Results

Brief summary of model performance metrics for each attribute:

| Attribute | Metric | Value |
|-----------|--------|-------|
| Age       | MAE    | TBD   |
| Gender    | F1     | TBD   |
| Accent    | F1     | TBD   |
| Text      | WER    | TBD   |

For detailed analysis and visualizations, refer to the notebooks in the `notebooks/` directory.

## 📚 References

- Mozilla Common Voice Dataset [Link]
- MFCC Feature Extraction [Link]
- Deep Learning for Speech Processing [Link]

## 🤝 Contributors

We want to thank the following individuals who have contributed to this project:

| Name | GitHub Username |
|------|-----------------|
| Roshan Roy Suja | roshan-rs-git |
| Simran Kapoor | simrankapoor456 |
| Anavi Reddy | Anavireddy404 |
| Monish Patalay | monishpatalay |
| Samprat Sakhare | samprat49 |
| Lynn Lee | lynnlee |
| Jooeun Jeon | roeldartz |

## 📄 License

This project uses the Mozilla Common Voice dataset, which is made available under the [CC-0 license](https://creativecommons.org/publicdomain/zero/1.0/). Our implementation code is available under the MIT License.
