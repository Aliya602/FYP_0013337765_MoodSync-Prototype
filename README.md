# FYP_0013337765_MoodSync-Prototype
Comparing Rule-Based and ML Models for Cross-Domain Music-to-Film Recommendation

**Overview**
The prototype,'MoodSync', compares rule-based and ML approaches/models for mapping music mood to film recommendations

**Features**
- -Rule-based classifier (using Russell's Circumplex Model)
- 5 ML models (Logistic Regression, Decision Tree, Random Forest, KNN and SVM)
- Flask backend API -Streamlit fronted UI
- Model comparison framework

**Setup Instructions**
-  pip install -r requirements.txt and other ones that are necessary
-  Download datasets from Kaggle (see below) Place datasets in 'data/' folder
-  Run Flask backend first: python src/app.py
-  Run Streamlit frontend on a new terminal: streamlit run src/ui.py

**Datasets (Download Separately)**
- Spotify Dataset: https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-1921-2020-160k-tracks
- Movie Dataset: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

**Results**
- Random Forest:99% F1 Score
- Rule-based baseline: 100% accuracy (synthetic labels) -Response time: 50ms
  
**Respository Structure**
.
├── src/
├── ├── data_loader.py # Data loading (loading the datasets)
├── ├── mood_classifier.py # Rule-based classifier
├── ├── mood_mapper.py # Mood-to-genre mapping
├── ├── app.py # Flask backend
├── ├── ui.py # Streamlit frontend
├── ├── model_comparison.py # ML model comparison
└── └── run_evaluation.py # Evaluation script

Make sure that all codes are under one file including the datasets

