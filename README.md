# Multi-Attribute Speech Analysis System

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
- **Feature engineering**: Extraction of 40 MFCC coefficients (mean and std), spectral features, and energy metrics
- **Deep learning models**: Specialized neural network architectures for each prediction task
- **Comprehensive evaluation**: Task-specific metrics to evaluate model performance

## 📂 Repository Structure

* `data/`: Raw and processed datasets
  * `raw/`: Original CSV files from Common Voice dataset (FinalDataset_sample.csv)
  * `processed/`: Preprocessed data ready for model training (accentdf.csv, agedf.csv, genderdf.csv, textdf.csv)
  * `Datacard.md`: Dataset documentation (Google Data Card)
* `notebooks/`: Jupyter Notebooks for data validation and analysis
  * `Data_visualization/`: Notebooks for data visualization (Data_Visualizations.ipynb)
  * `feature_extraction/`: Notebooks for feature extraction (audio_feature_extraction_final.ipynb)
  * `final_model_with_prediction/`: Notebooks for model implementation (Final_Model_with_Prediction.ipynb)
  * `preprocessing/`: Notebooks for data preprocessing (Final_feature_prep.ipynb)
* `src/`: README.txt explaining about the dataset
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

Summary of model performance metrics for each attribute:

| Attribute | Accuracy | F1 Score | WER   |
|-----------|----------|----------|-------|
| Age       | 92%      | 92.2%    | -     |
| Gender    | 98%      | 98%      | -     |
| Accent    | 92%      | 92%      | -     |
| Text      | -        | -        | 8.06% |

For detailed analysis and visualizations, refer to the notebooks in the `notebooks/` directory, particularly the `Final_Model_with_Prediction.ipynb` file in the `final_model_with_prediction` folder.

## 👥 Team Members & Roles

| Name | GitHub Username | Role & Responsibilities |
|------|-----------------|-------------------------|
| Simran Kapoor | simrankapoor456 | **Project Lead / Coordinator**: Manages overall project direction, coordinates team efforts, and ensures deliverables meet deadlines. |
| Samprat Sakhare | samprat49 | **Data Acquisition & Cleaning Lead**: Handles dataset processing, cleaning, and preparation for feature extraction. |
| Monish Patalay | monishpatalay | **Audio Feature Engineering Specialist**: Implements audio feature extraction pipelines and optimizes MFCC configurations. |
| Anavi Reddy | Anavireddy404 | **Model Architect (Age & Gender)**: Develops and trains neural networks for age and gender prediction tasks. |
| Roshan Roy Suja | roshan-rs-git | **Model Architect (Accent & Text)**: Builds models for accent classification and speech-to-text transcription. |
| Lynn Lee | lynnlee128 | **Evaluation & Metrics Lead**: Designs evaluation frameworks and analyzes model performance across attributes. |
| Jooeun Jeon | roeldartz | **Documentation & Reporting Lead**: Creates project documentation and prepares final deliverables. |

## 📚 References

- [Mozilla Common Voice Dataset](https://www.kaggle.com/datasets/mozillaorg/common-voice/data)
- [MFCC Feature Extraction](https://pytorch.org/audio/main/generated/torchaudio.transforms.MFCC.html)
- [Deep Learning for Speech Processing](https://www.deeplearningbook.org/)
- [Common Voice Corpus](https://commonvoice.mozilla.org/en/datasets)

## 📄 License

This project uses the Mozilla Common Voice dataset. Mozilla website content is available under the [Creative Commons Attribution Share-Alike License v3.0](https://www.mozilla.org/en-US/foundation/licensing/website-content/) or any later version.

Our implementation code is available under the MIT License.

For more information about Mozilla's licensing policies, please visit their [Licensing Policies page](https://www.mozilla.org/en-US/foundation/licensing/).
