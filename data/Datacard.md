# Multi-Attribute-Speech-Analysis-System

Using a multi-task deep learning architecture, we aim to develop a comprehensive speech analysis system that simultaneously extracts demographic information, accent features, and text content from audio input, enabling more personalized and efficient voice-based interactions across industries.

#### Dataset Link
https://www.kaggle.com/datasets/mozillaorg/common-voice/data

#### Data Card Author(s)
- **Simran Kapoor**
- **Roshan Roy Suja**
- **Anavi Reddy**
- **Monish Patalay**
- **Samprat Sakhare**
- **Jooeun Jeon**
- **Lynn Lee**

## 1. Authorship
### Publishers
#### Publishing Organization(s)
Mozilla

#### Industry Type(s)
- Corporate - Tech

#### Contact Detail(s)
- **Publishing POC:** Mozilla Foundation
- **Affiliation:** Mozilla
- **Website:** https://commonvoice.mozilla.org/en/datasets

## 2. Dataset Overview
#### Data Subject(s)
- Non-Sensitive Data about people

#### Dataset Snapshot
Category | Data
--- | ---
Size of Dataset | 13 GB
Number of Instances | 380,368 rows
Number of Fields | 7 base + extracted features
Labeled Classes | Age, gender, accent categories
Number of Labels | Variable per attribute

**Additional Notes:** After feature extraction, we have multiple MFCC features (13 mean and std values), spectral features, and RMS values in addition to the base fields.

#### Content Description
Common Voice is a collection of speech data read by users on the Common Voice website, based on text from various public sources. The dataset contains the transcription of each audio clip and the age, gender, and accent of each speaker, among other features. Its primary purpose is to enable the training and testing of automatic speech recognition (ASR) systems.

#### Descriptive Statistics
After feature extraction, the dataset includes columns like filename, text, up_votes, down_votes, age, gender, accent, duration, mel_energy_mean, mel_energy_std, mfcc1_mean through mfcc13_mean, mfcc1_std through mfcc13_std, spectral_centroid_mean, spectral_centroid_std, rms_mean, and rms_std.

### Sensitivity of Data
#### Sensitivity Type(s)
- None

#### Risk Type(s)
- No Known Risks

### Dataset Version and Maintenance
#### Maintenance Status
**Limited Maintenance** - The data will not be updated, but any technical issues will be addressed.

#### Version Details
**Current Version:** 1.0
**Last Updated:** 11/2017
**Release Date:** 11/2017

## 3. Example of Data Points
#### Primary Data Modality
- Audio Data

#### Sampling of Data Points
- Demo Link: https://commonvoice.mozilla.org/en/datasets
- Typical Data Point Link: https://github.com/roshan-rs-git/cs5661-dsproject-Multi-Attribute-Speech-Analysis-System/blob/main/notebooks/gender.ipynb

The primary data modality is audio data, supported by structured tabular metadata and transcribed text. Each data point consists of:
- An .mp3 file containing the recorded voice of a speaker
- A transcription (text) of the audio
- Demographic labels (age, gender, accent)
- Metadata (up_votes, down_votes, duration)

This multimodal structure enables diverse modeling approaches, including classification (gender, accent), regression (age group estimation), and sequence prediction (transcription).

To ensure data quality and balanced representation across tasks, the following sampling techniques were applied:
- **Filtering Rules**:
  - For general preprocessing: retained only rows where up_votes ≥ down_votes
  - For the invalid.tsv file: applied a stricter threshold of up_votes > down_votes
- **Task-Specific Sampling**:
  - Speech-to-Text: Focused on high-agreement clips using both text and audio features
  - Gender Prediction: Retained entries with non-missing gender values
  - Age/Accent Prediction: Focused on rows where the target label was present

#### Data Fields
Field Name | Field Value | Description
--- | --- | ---
path | cv-valid-train/sample-000001.mp3 | Path to audio clip
text | "everything in the universe evolved he said" | Transcription of the spoken sentence
age | 30s | Reported age group of the speaker
gender | male | Reported gender
accent | us | Reported accent/dialect
up_votes | 5 | Number of community upvotes
down_votes | 0 | Number of community downvotes
duration | 3.2 | Length of the audio clip in seconds
mfcc_1_mean | -456.23 | Mean of MFCC coefficient 1
spectral_centroid | 2340.45 | Spectral brightness of the audio
rms_energy | 0.0052 | RMS energy of the signal

**Caption:** Example field values and their corresponding descriptions used in model training.
**Additional Notes:** MFCC features (13 mean and std values) and spectral features were derived from PyTorchAudio transformations.

#### Typical Data Point
```
{
  "path": "cv-valid-train/sample-000001.mp3",
  "text": "everything in the universe evolved he said",
  "age": "30s",
  "gender": "male",
  "accent": "us",
  "up_votes": 5,
  "down_votes": 0,
  "duration": 3.2,
  "mfcc_1_mean": -456.23,
  "mfcc_1_std": 18.56,
  "spectral_centroid_mean": 2340.45,
  "rms_energy_mean": 0.0052
}
```

**Additional Notes:** This example is considered typical as it includes complete, consistent metadata and strong community validation. It is usable for all modeling tasks.

#### Atypical Data Point
```
{
  "path": "cv-valid-train/sample-5678.mp3",
  "text": "",
  "age": "",
  "gender": "",
  "accent": "",
  "up_votes": 0,
  "down_votes": 3,
  "duration": 1.1
}
```

**Additional Notes:** This data point is atypical due to missing text and demographic labels, and because the community flagged the clip as problematic. Such entries were excluded from our preprocessing pipeline.

## 4. Motivations & Intentions
### Motivations
#### Purpose(s)
- Research

#### Domain(s) of Application
Automatic Speech Recognition (ASR), Speaker Demographic Modeling, Speech-based Personalization, Accent Classification, Speech Feature Analysis in AI Education

#### Motivating Factor(s)
- **Representation Gaps:** Many commercial ASR systems are biased toward majority accents/demographics. We explore whether open-source data can be leveraged for fairer modeling.
- **Multi-Task Speech Learning:** Investigating the feasibility of using one dataset to perform multiple voice-related tasks.
- **Open Research:** Building with Common Voice demonstrates how public datasets can drive inclusive voice AI.
- Mozilla's Common Voice provides a platform for voice diversity and openness: https://commonvoice.mozilla.org/en/about

### Intended Use
#### Dataset Use(s)
- Safe for research use

#### Suitable Use Case(s)
**Suitable Use Case:** Training and evaluating open-source ASR models
**Suitable Use Case:** Benchmarking demographic prediction models
**Suitable Use Case:** Researching fairness, bias, and representational ethics in speech data
**Suitable Use Case:** Teaching audio processing and deep learning workflows

**Additional Notes:** Designed for reproducible experimentation using open-source tools

#### Unsuitable Use Case(s)
**Unsuitable Use Case:** Surveillance or identification of individuals using voice patterns
**Unsuitable Use Case:** Commercial profiling based on gender, age, or accent without user consent
**Unsuitable Use Case:** Deploying speech models trained on this dataset in real-world production systems without fairness evaluation

**Additional Notes:** Dataset is not intended for biometric, forensic, or sensitive decision-making use cases.

#### Research and Problem Space(s)
This dataset supports inquiry into the following research areas:
- Bias and fairness in speech modeling
- Accent and linguistic variation in machine learning
- Low-resource demographic classification
- Robustness of ASR under varied speech quality
- Multi-task learning with shared feature space (MFCCs)

#### Citation Guidelines
**Guidelines & Steps:** Please cite Mozilla Common Voice and our GitHub project when using the data or code:
```
@dataset{mozilla_common_voice,
  title={Mozilla Common Voice Dataset},
  author={Mozilla Foundation},
  year={2020},
  url={https://commonvoice.mozilla.org/en/datasets}
}
```

GitHub Project: https://github.com/roshan-rs-git/cs5661-dsproject-Multi-Attribute-Speech-Analysis-System

## 5. Access, Retention, & Wipeout
### Access
#### Access Type
- External - Open Access

#### Documentation Links
- Common Voice Dataset Docs: https://commonvoice.mozilla.org/en/datasets
- Kaggle Dataset Page: https://www.kaggle.com/datasets/mozillaorg/common-voice
- GitHub: https://github.com/mozilla/voice-web
- GitHub: https://github.com/common-voice/common-voice

#### Prerequisites
None

#### Policy Links
- Mozilla Data License: https://github.com/common-voice/common-voice/blob/main/LICENSE
- Kaggle Download: https://www.kaggle.com/datasets/mozillaorg/common-voice/data
- Common Voice Dataset Mirror: https://voice.mozilla.org/en/datasets

Code to download data:
```
!kaggle datasets download -d mozillaorg/common-voice
```

#### Access Control List(s)
**ACL:** Public access via Kaggle; no special permissions required.
**ACL:** Users must agree to Mozilla's open license.
**ACL:** Contributions to dataset are moderated by Mozilla to maintain data quality.

**Additional Notes:** Dataset includes voice samples and metadata (e.g., gender, age, accent). No sensitive identifiers present.

### Retention
#### Duration
Indefinite

#### Policy Summary
**Retention Plan ID:** MOZ-CV-OPEN-2025
**Summary:** Mozilla retains the dataset indefinitely with periodic updates and releases.

#### Process Guide
This dataset follows Mozilla's open data standards and Creative Commons licensing.

#### Exception(s) and Exemption(s)
**Exemption Code:** PUBLIC_DATA
**Summary:** No retention limits; dataset is openly available under CC-0 license.

**Additional Notes:** Mozilla may remove or modify content based on data quality or privacy concerns.

### Wipeout and Deletion
#### Duration
Not applicable

#### Deletion Event Summary
Not applicable

#### Acceptable Means of Deletion
- Manual removal from local storage
- Deletion of downloaded zip or extracted content
- Unlinking from version control mirrors or dataset references

#### Post-Deletion Obligations
Not applicable

## 6. Provenance
### Collection
#### Method(s) Used
- Crowdsourced - Volunteer
- Taken from other existing datasets

#### Methodology Detail(s)
**Collection Type:** Open-source, multilingual voice dataset for training and evaluating speech recognition systems.

**Source:** Common Voice Dataset on Kaggle: https://www.kaggle.com/datasets/mozillaorg/common-voice/data

**Platform:** Kaggle
**Description:** Kaggle hosts the downloadable .tsv metadata and audio .mp3 files by language and version.
**URL:** https://www.kaggle.com/datasets/mozillaorg/common-voice

**Is this source considered sensitive or high-risk?** No

**Dates of Collection:** Language-specific; dataset version reflects a snapshot up to late 2022.

**Primary modality of collection data:**
- Audio Data
- Text Data
- Tabular Data

**Update Frequency for collected data:** Static

**Additional Links for this collection:**
- https://github.com/common-voice/common-voice/blob/main/LICENSE

#### Source Description(s)
https://www.kaggle.com/datasets/mozillaorg/common-voice
Includes zipped folders for each language. Each folder contains .tsv metadata files and .mp3 voice clips.

#### Collection Cadence
Static: Data was collected once from single or multiple sources.

#### Data Integration
Kaggle-hosted Common Voice dataset zip files.

#### Data fields that were collected and are included in the dataset
Field Name | Description
--- | ---
client_id | Anonymous contributor identifier
path | Relative path to the .mp3 file
sentence | Prompted sentence spoken by the contributor
up_votes | Number of positive community validations
down_votes | Number of negative validations
age | Reported age range of speaker
gender | Reported gender of speaker
accent | Optional field identifying accent
locale | Language locale

**Additional Notes:** Not all speakers provide age, gender, or accent.

#### Collection Method or Source
**Description:** Data was originally collected through the Common Voice web platform, then curated and released in versioned form. Kaggle hosts versioned zip files per language.
**Methods employed:** Clip validation via community voting
**Tools or libraries:** Mozilla's internal scripts for preprocessing

### Collection Criteria
#### Data Selection
Only voice clips with at least one upvote are included in validated.tsv, train.tsv and test.tsv are created from validated data

#### Data Inclusion
- Contributors agree to CC0 licensing
- Metadata fields are self-reported and voluntary

#### Data Exclusion
No names, email addresses, or raw recordings outside CC0 license are present

### Relationship to Source
#### Use & Utility(ies)
**Source Type:** Open-access dataset hosted on Kaggle
**Purpose:** Training and testing automatic speech recognition (ASR) models
**Link:** https://www.kaggle.com/datasets/mozillaorg/common-voice

#### Benefit and Value(s)
- Public domain licensing (CC0)
- High diversity in language and demographics

#### Limitation(s) and Trade-Off(s)
- Some recordings may have background noise
- Some languages have small sample sizes

### Version and Maintenance
#### First Version
**Release date:** Approx. 11/2017 (based on latest Kaggle upload)
**Link to dataset:** Mozilla Common Voice (Kaggle): https://www.kaggle.com/datasets/mozillaorg/common-voice
**Status:** Static (Not actively maintained on Kaggle)
**Size of Dataset:** ~13 GB zipped
**Number of Instances:** ~300,000+ clips across full dataset
**Note(s) and Caveat(s):** Each Kaggle version is a fixed snapshot and may not reflect latest Mozilla Common Voice updates.
**Cadence:** Static

#### Last and Next Update(s)
**Date of last update:** 15/11/2019
**Total data points affected:** ~300,000+
**Data points updated:** N/A
**Data points added:** Full new version upload
**Data points removed:** None
**Date of next update:** Unknown

## 7. Human and Sensitive Attributes
#### Sensitive Human Attribute(s)
- Gender
- Age
- Geography
- Language

#### Intentionality
**Intentionally Collected Attributes**

Field Name | Description
--- | ---
age | Speaker's age
gender | Speaker's gender
accent | Speaker's geographic/linguistic accent
language | Language spoken (English)

#### Known Correlations & Risks
**Potential Correlations:**
- age ↔ gender
- accent ↔ geography or ethnicity
- upvotes/downvotes ↔ possible bias in user ratings

#### Risks & Mitigations
Risk | Description | Mitigation
--- | --- | ---
Bias in Labels | Some age/gender/accent labels may be self-reported or incomplete | Dropped NaN rows, performed label balance checks
Representation Bias | Dataset may not equally represent all accents or age groups | Acknowledge in model limitations; stratified sampling if needed
Inferring Sensitive Info | Model could infer identity attributes | Used features only for defined scope, anonymized audio metadata

## 8. Extended Use
### Use with Other Data
**Safety Level:** Safe to use with other data

**Description:** The Mozilla Common Voice dataset is designed to be open, modular, and compatible with various other speech/audio datasets. It includes audio clips and corresponding transcriptions across multiple languages, accents, and demographics, making it highly suitable for use in multimodal and multilingual training pipelines.

### Known Safe Dataset(s) or Data Type(s)
- **LibriSpeech:** A widely-used English corpus derived from audiobooks; compatible due to similar format (audio + text) and open licensing
- **TED-LIUM:** Contains transcribed TED Talks; shares structural similarity and often used in transfer learning with Common Voice
- **Google Speech Commands Dataset:** Contains short spoken commands; though simpler, it's useful for pretraining or finetuning with Common Voice for specific tasks

### Best Practices
- Normalize sampling rates across datasets
- Match speaker metadata for demographic-specific studies
- Use language tags when merging multilingual datasets
- Evaluate model performance separately on each dataset to detect distributional shifts

**Additional Notes:**
- Ensure license compatibility when merging with proprietary or restricted datasets
- Always cite Mozilla and contributing users when using Common Voice

### Known Unsafe Dataset(s) or Data Type(s)
- Private audio datasets without user consent: Legal and ethical risks due to different consent frameworks
- Noisy datasets with poor transcriptions: May degrade performance when combined without cleaning

### Limitation(s) and Recommendation(s)
- **Limitation:** Underrepresentation of certain languages or dialects
  - **Recommendation:** Use sampling or data augmentation to balance for language bias
- **Limitation:** Clip length and speaking pace vary significantly across datasets
  - **Recommendation:** Apply temporal normalization and silence trimming during preprocessing

### Forking & Sampling
**Safety Level:** Safe to fork and/or sample

### Acceptable Sampling Method(s)
- Random Sampling
- Stratified Sampling (e.g., by gender, language, or accent)
- Weighted Sampling (to balance underrepresented groups)

### Best Practices for Sampling
- Maintain balanced distribution of demographics and languages when subsetting
- Avoid sampling only by clip length or speaker ID to prevent induced bias
- Use metadata such as age, gender, and accent for stratified sampling

## 9. Transformations
### Synopsis
#### Transformation(s) Applied
- Cleaning Missing Values
- Converting Data Types
- Data Aggregation
- Feature Extraction

### Breakdown of Transformations
#### Data Cleaning and Filtering
- Removed rows with missing or mismatched labels
- Filtered text data with upvotes >= downvotes for reliability
- Converted categorical labels (gender, age groups) to integer encodings

#### Feature Engineering
- Applied MFCC extraction using multiple configurations
- Calculated spectral and RMS values
- Normalized features to ensure consistent scale across models

#### Tools and Libraries Used
Tool / Library | Purpose
--- | ---
torchaudio | Audio processing and feature extraction
torch | Model definition and training
pandas, numpy | Data manipulation
sklearn | Label encoding, data splitting, evaluation metrics
matplotlib, seaborn | Visualizations

#### Comparative Summary
Stage | Before | After
--- | --- | ---
Raw Data | 380,368 rows | Filtered per task (~120,000 to 350,000)
Columns | 8 | Audio features + target
MFCCs | None | 13–40 MFCCs + spectral features
Text Quality | Mixed | Filtered by upvote/downvote score

#### Human Oversight Measures
- Performed manual review of preprocessing logic
- Validated label encoding schemes
- Visualized class distributions to detect imbalances
- Conducted multiple experiments to verify MFCC efficacy

#### Residual Risks and Limitations
- Accent and gender recognition may be biased due to underrepresentation
- Transcription quality is user-rated and not fully verified
- Audio clips may vary significantly in background noise and recording quality

#### Additional Considerations
- Mozilla's dataset is openly licensed and supports reproducible research
- Our system is intended for academic purposes and not deployed in production
- All models are trained with transparency and fairness in mind

## 10. Annotations & Labeling
#### Annotation Workforce Type
- Annotation Target: Speech-to-text transcription
- Machine-Generated Annotations: No
- Human Annotations (Non-Expert): Yes
- Human Annotations (Crowdsourcing): Yes
- Other Sources: Open-source contributors from Mozilla Common Voice community
- Unlabeled Data: No

#### Annotation Characteristics
- Number of Unique Annotations: Varies by language
- Total Number of Annotations: Millions (across all languages)
- Average Annotations per Example: About 1
- Number of Annotators per Example: Typically 1–2
- Quality Metric: Varies per language (e.g., Word Error Rate)
- Validation Ratio: Approximately 60–70% of clips are validated

#### Annotation Descriptions
**(Transcription Annotations)**

**Description:** Contributors listen to audio clips and manually type the exact spoken words (without punctuation). These are then validated by other users.

**Link:** Mozilla Common Voice

**Platforms, tools, or libraries:**
- Mozilla Common Voice Web App (for recording, transcription, and validation)
- Simple web-based tools for collecting and checking clips

**Additional Notes:** Community-specific transcription guidelines may apply per language

#### Annotation Distribution
**(Transcription Annotations)**
- Correct/Validated Transcriptions: Approx. 60–70%
- Flagged or Invalid Transcriptions: 10–20%
- Unvalidated Clips: 10–30%

**Additional Notes:** Distribution varies by language depending on community participation

#### Annotation Task(s)
**(Transcription Task)**

**Task description:** Transcribe what's heard in each clip or validate others' transcriptions

**Task instructions:**
- Transcribe exactly as heard (no punctuation)
- Mark as incorrect if mismatched

**Methods used:** Manual entry and peer review

**Inter-rater adjudication policy:** Based on community voting (multiple validators)

**Golden questions:** Not formally used, though consistency is emphasized

**Additional notes:** Annotation quality is influenced by contributor engagement per language

### Human Annotators
#### Annotator Description(s)
**(Transcription Annotations)**

**Task type:** Transcription and validation

**Number of unique annotators:** Tens of thousands globally

**Expertise of annotators:** Non-expert, volunteer-based

**Description of annotators:** Open to all users

**Language distribution of annotators:** Over 100 languages

**Geographic distribution of annotators:** Global

**Summary of annotation instructions:** Language-specific instructions available on the Common Voice website

**Summary of gold questions:** Not implemented formally

**Annotation platforms:** Mozilla Common Voice website

**Additional Notes:** Driven by an open and inclusive contributor community

#### Languages
**(Transcription Annotations)**
- English: ~35%
- German: ~10%
- French: ~8%
- Spanish: ~7%
- Other (90+ languages): ~40%

#### Locations
**(Transcription Annotations)**
- North America: ~35%
- Europe: ~30%
- Asia: ~15%
- Africa & South America: ~10%
- Other/Unknown: ~10%

#### Genders
**(Transcription Annotations)**
- Male: ~55%
- Female: ~40%
- Other/Unspecified: ~5%

## 11. Validation Types
#### Method(s)
- Data Type Validation
- Range and Constraint Validation
- Structured Validation
- Consistency Validation

#### Breakdown(s)
**(Community-Based Verification)**

**Number of Data Points Validated:** Millions (language-dependent)

**Fields Validated**
Field |
--- |
audio_clip |
transcription |
language |

#### Description(s)
**(Community-Based Verification)**

**Method:** Users record voice clips and provide transcriptions. Other contributors listen and vote on transcription accuracy.

**Platforms, tools, or libraries:**
- Common Voice Web UI
- Mozilla backend systems
- PostgreSQL and metadata checking scripts

**Validation Results:** ~60–70% of clips are validated. Invalid clips are flagged and excluded. Some languages have higher validation rates depending on community size.

**Additional Notes:** Validation is open and transparent. No automated adjudication is applied.

### Description of Human Validators
#### Characteristic(s)
**(Human crowdsource validation)**
- Unique validators: Tens of thousands worldwide
- Number of examples per validator: Varies widely (some >100, some only a few)
- Average cost/task/validator: $0 (volunteer-based)
- Training provided: No
- Expertise required: No

#### Description(s)
**(Human crowdsource validation)**

**Validator description:** Validators are volunteers who mark transcriptions as valid or invalid.

**Training provided:** No formal training; instructions are provided on the website.

**Validator selection criteria:** No selection criteria; open to all volunteers

**Additional Notes:** The community-driven approach allows for diverse validation

#### Languages
**(Validation Coverage)**
- English: ~35%
- German: ~10%
- French: ~8%
- Spanish: ~7%
- Other (100+ languages): ~40%

**Note:** Reflects the diversity of the Common Voice dataset.

#### Locations
**(Validation Contributors)**
- North America: ~35%
- Europe: ~30%
- Asia: ~15%
- Africa & South America: ~10%
- Other/Unknown: ~10%

**Note:** Global contributor base supports language and accent diversity.

#### Genders
**(Validation Metadata)**
- Male: ~55%
- Female: ~40%
- Other/Unspecified: ~5%

**Note:** Gender is optional metadata; not all clips include this info.

## 12. Sampling Methods
#### Method(s) Used
- Random Sampling
- Stratified Sampling
- Weighted Sampling
- Other: Community-driven contribution rather than systematic sampling

#### Characteristic(s)
**(Community-driven sampling)**
- Upstream Source: Global volunteer base
- Total Data Sampled: Over 20,000 hours of audio
- Sample Size: Varies by task/model
- Sampling Rate: Language-dependent
- Statistics: Usually not precomputed
- Sampling Variation: High across languages
- Sampling Statistics: Covers a wide range of accents, age groups, and languages

#### Sampling Criteria
- Typically random or language-specific depending on model goals
- Stratified sampling is used for benchmarking to ensure class balance
- No enforced demographic balance, but metadata enables post-hoc filtering

## 13. Known Applications & Benchmarks
#### ML Application(s)
Automatic Speech Recognition (ASR), Accent Classification, Multilingual Fine-Tuning, Keyword Spotting

#### Evaluation Results
**(Various Speech Recognition Models)**

**Model Card:** Example — Whisper Model Card

Evaluation Results
- Accuracy / WER: Varies by model and language
- Precision/Recall: Evaluated using Word Error Rate (WER)
- Performance Metrics:
  - Word Error Rate (WER)
  - Character Error Rate (CER)
  - Latency

#### Evaluation Process(es)
**(Fine-tuning Method)**

**Method:** Fine-tune on Common Voice, evaluate on held-out samples

**Steps:**
- Preprocessing: Normalize audio, trim silence, tokenize
- Training: Use pretrained ASR models
- Evaluation: Compare transcriptions to ground truth

**Influencing Factors:**
- Clip length
- Speaker accent
- Background noise
- Microphone quality

**Considerations:**
- Multilingual processing
- Speaker diversity
- Audio quality variability

**Results:** Found in model papers and Hugging Face model cards

#### Description(s) and Statistic(s)
**(Common Speech Recognition Models)**

**Model Description:** Various pre-trained and fine-tuned speech recognition models

- Models: Wav2Vec2, Whisper, DeepSpeech
- Model Size: Hundreds of millions of parameters
- Weights: Provided by original training sources
- Latency: Depends on model architecture and deployment

#### Expected Performance and Known Caveats
**(Common Speech Recognition Models)**

**Expected Performance:** Best for high-resource languages (e.g., English, French)

**Known Caveats:**
- Performance drops for low-resource languages
- Accuracy impacted by poor recording quality or strong accents
- Potential demographic bias if metadata is not filtered or balanced

## 14. Terms of Art
### Concepts and Definitions referenced in this Data Card

#### Mel-frequency Cepstral Coefficients (MFCC)
Definition: A representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. MFCCs are commonly used as features in speech and speaker recognition systems.

Source: Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357-366. https://doi.org/10.1109/TASSP.1980.1163420

Interpretation: In our project, MFCCs are the primary features extracted from speech audio files to capture the unique characteristics of speech that help in predicting age, gender, accent, and transcribed text. We extract both mean and standard deviation values for 13 MFCC coefficients.

#### Spectral Centroid
Definition: A measure used in digital signal processing to characterize a spectrum. It indicates where the "center of mass" of the spectrum is located and is calculated as the weighted mean of the frequencies present in the signal, with their magnitudes as weights.

Source: Peeters, G. (2004). A large set of audio features for sound description (similarity and classification) in the CUIDADO project. CUIDADO IST Project Report, 54(0), 1-25.

Interpretation: In our speech analysis system, spectral centroid features help capture the brightness or clarity of a speaker's voice, which can vary based on age, gender, and accent. We use both mean and standard deviation of the spectral centroid.

#### Root Mean Square (RMS)
Definition: A statistical measure of the magnitude of a varying quantity, calculated as the square root of the mean of the squares of the values. In audio processing, RMS is used to measure the average power of an audio signal.

Source: Lerch, A. (2012). An Introduction to Audio Content Analysis: Applications in Signal Processing and Music Informatics. Wiley-IEEE Press. https://doi.org/10.1002/9781118393550

Interpretation: In our system, RMS features help capture the overall energy and volume characteristics of speech, which can vary based on speaker demographics. We extract both mean and standard deviation of RMS values.

#### Mel Energy
Definition: The energy measurement of an audio signal after it has been transformed to the mel scale, which is a perceptual scale of pitches judged by listeners to be equal in distance from one another.

Source: Stevens, S. S., Volkmann, J., & Newman, E. B. (1937). A scale for the measurement of the psychological magnitude pitch. The Journal of the Acoustical Society of America, 8(3), 185-190. https://doi.org/10.1121/1.1915893

Interpretation: Mel energy features in our system help capture the perceptual energy distribution across the frequency spectrum of speech, which varies significantly between different demographic groups and can help in accent and gender identification.

#### Common Voice Dataset
Definition: An open-source dataset of voiced recordings created by Mozilla, designed to help train and test machine learning algorithms for voice recognition.

Source: Ardila, R., et al. (2020). Common Voice: A Massively-Multilingual Speech Corpus. Proceedings of the 12th Language Resources and Evaluation Conference, 4218-4222. https://aclanthology.org/2020.lrec-1.520/

Interpretation: This is the primary dataset used in our Multi-Attribute-Speech-Analysis-System, providing audio samples along with metadata including speaker age, gender, accent, and transcribed text, allowing for supervised learning of multiple attributes from speech.

#### Multi-task Learning
Definition: A machine learning approach where a single model is trained to perform multiple related tasks simultaneously, often leading to improved generalization performance compared to training separate models for each task.

Source: Caruana, R. (1997). Multitask Learning. Machine Learning, 28(1), 41-75. https://doi.org/10.1023/A:1007379606734

Interpretation: While our current approach uses separate neural networks for each attribute prediction task, multi-task learning represents a potential future direction where a shared base network with specialized heads could improve overall performance through learned shared representations.

#### Feature Extraction
Definition: The process of transforming raw data into a set of features that can be effectively processed by machine learning algorithms while still representing the original data with sufficient accuracy.

Source: Zheng, A., & Casari, A. (2018). Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists. O'Reilly Media.

Interpretation: In our system, feature extraction involves transforming raw audio waveforms into meaningful representations (MFCCs, spectral features, etc.) that capture relevant characteristics for predicting age, gender, accent, and text.

#### Deep Neural Network (DNN)
Definition: A class of machine learning algorithms that use multiple layers of interconnected nodes (neurons) to model complex patterns in data and make predictions or classifications.

Source: Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. https://www.deeplearningbook.org/

Interpretation: We use separate DNNs to predict each attribute (age, gender, accent, and text) from the extracted audio features, with architectures tailored to the specific classification or regression task.

#### Word Error Rate (WER)
Definition: A common metric for evaluating automatic speech recognition systems, calculated as the sum of substitution, insertion, and deletion errors divided by the total number of words in the reference.

Source: McCowan, I. A., et al. (2004). On the use of information retrieval measures for speech recognition evaluation. IDIAP Research Report, IDIAP-RR-04-73.

Interpretation: WER serves as an evaluation metric for the text prediction component of our system, measuring how accurately the model can transcribe spoken content from audio features.

#### Z-score Normalization
Definition: A data preprocessing technique that transforms data to have a mean of 0 and a standard deviation of 1, calculated by subtracting the mean and dividing by the standard deviation.

Source: Patro, S. G. K., & Sahu, K. K. (2015). Normalization: A Preprocessing Stage. IARJSET, 2(3), 20-22. https://doi.org/10.17148/IARJSET.2015.2305

Interpretation: We apply z-score normalization to our extracted audio features to ensure all features contribute equally to model training regardless of their original scales, improving convergence and performance of our neural networks.

#### F1-Score
Definition: A measure of a model's accuracy that considers both precision and recall, calculated as their harmonic mean. It ranges from 0 to 1, with higher values indicating better performance.

Source: Powers, D. M. W. (2011). Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation. Journal of Machine Learning Technologies, 2(1), 37-63.

Interpretation: F1-score serves as an evaluation metric for our classification tasks (gender and accent prediction), providing a balanced measure of the model's ability to correctly identify classes while minimizing false positives and false negatives.

## 15. Reflections on Data

### Data Imbalance Considerations
The Common Voice dataset exhibits imbalances in demographic representation, with certain age groups, genders, and accents having significantly more samples than others. This imbalance can lead to biased models that perform better on well-represented groups. We address this through stratified sampling during training/validation splits and by implementing class weights in our loss functions to give more importance to underrepresented classes during training.

### Audio Quality Variation
The dataset contains recordings from diverse sources with varying audio quality, microphone types, and background noise levels. These variations can affect feature extraction and model performance. Our preprocessing pipeline includes normalization of audio features to mitigate these effects, but future work could incorporate more robust features or data augmentation techniques to further address quality inconsistencies.

### Ethical Considerations in Demographic Prediction
Developing systems that predict personal attributes such as age and gender from voice raises important ethical considerations regarding privacy, consent, and potential misuse. While our system aims to demonstrate technical capabilities in multi-attribute prediction, real-world applications should implement appropriate safeguards, obtain proper consent, and consider potential discriminatory impacts. The system should be developed and deployed responsibly with transparency about its capabilities and limitations.

### Feature Selection Methodology
Our current approach of testing different numbers of MFCC coefficients through trial and error is computationally expensive and may not yield optimal results. Future iterations could implement more systematic feature selection techniques such as mutual information analysis, recursive feature elimination, or principal component analysis to identify the most relevant features for each prediction task, potentially improving model performance while reducing computational requirements.