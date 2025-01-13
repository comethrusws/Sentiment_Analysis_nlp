# IMDB Sentiment Analysis (Logistic regression and transformers method)

## Project Overview
This project is a submission for the Fellowship.ai Cohort 33 challenge. It implements sentiment analysis on the IMDB dataset using multiple approaches to demonstrate proficiency in natural language processing and machine learning.

## Challenge Description
The challenge involves building a sentiment analysis system that can effectively classify movie reviews as positive or negative, showcasing:
- Data preprocessing capabilities
- Feature engineering skills
- Model implementation and evaluation
- Code organization and documentation
- Use of modern NLP techniques (SpaCy and Transformers)

## Technical Implementation
Originally developed as a Jupyter Notebook (.ipynb) and converted to Python script, this solution implements:
- Data preprocessing using SpaCy with GPU acceleration
- TF-IDF vectorization for feature extraction
- Logistic Regression for baseline classification
- Transformer-based models for advanced sentiment analysis
- Comprehensive visualization and evaluation metrics

## Requirements
- Python 3.x
- Google Colab (for original notebook execution)
- Required packages:
  - pandas
  - spacy
  - scikit-learn
  - seaborn
  - matplotlib
  - transformers
  - kaggle
  - tqdm

## Project Structure
```
fellowship_sentiment_analysis_challenge_basab.ipynb/
├── Data Loading and Inspection
├── Text Preprocessing
│   ├── HTML cleanup
│   ├── Character normalization
│   └── SpaCy processing
├── Feature Engineering
│   └── TF-IDF Vectorization
├── Model Implementation
│   ├── Logistic Regression
│   └── Transformer-based analysis
└── Evaluation and Visualization
```

## Key Features

### Advanced Preprocessing
```python
def preprocess_text(text, nlp):
    text = clean_text(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and len(token.text) > 2]
    return ' '.join(tokens)
```

### GPU Acceleration
- Utilizes SpaCy's GPU capabilities for faster processing
- Optimized for Google Colab's GPU environment

### Model Pipeline
1. Data cleaning and preprocessing
2. Feature extraction using TF-IDF
3. Model training with Logistic Regression
4. Performance evaluation and visualization
5. Advanced sentiment analysis using Transformers

## Running the Project
1. Open the notebook in Google Colab
2. Upload your Kaggle credentials
3. Run all cells sequentially
4. Review the visualizations and performance metrics

## Implementation Details
- Uses SpaCy's `en_core_web_sm` model for preprocessing
- Implements TF-IDF vectorization with:
  - max_features: 50,000
  - ngram_range: (1, 2)
- Logistic Regression parameters:
  - C: 1.0
  - max_iterations: 1000
  - n_jobs: -1 (parallel processing)

## Results and Visualization
The project provides:
- Classification metrics
- Confusion matrix visualization
- Review length distribution analysis
- Sample predictions using transformer models

## Future Improvements
1. Implement cross-validation
2. Add more advanced preprocessing techniques
3. Experiment with different transformer architectures
4. Add model comparison metrics
5. Implement model serialization

## Notes
- This project was developed as part of the Fellowship.ai Cohort 33 application process
- Originally developed in Google Colab for GPU acceleration
- Focuses on demonstrating both traditional ML and modern NLP approaches
