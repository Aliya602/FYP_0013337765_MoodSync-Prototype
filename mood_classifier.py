from model_comparison import Comparison
from data_loader import load_Spotify_data, load_Movie_data
import pandas as pd
def classify_mood(valence, energy):
    # Classify mood based on Russell's circumplex model
    #Returns 'Happy', 'Sad', 'Energetic', or 'Calm'
    if valence >= 0.65 and energy >= 0.55:
        return 'Happy'
    elif valence < 0.35 and energy < 0.35:
        return 'Sad'
    elif energy < 0.70:
        return 'Energetic'
    else:
        return 'Calm'
    

# Determine mood from song features dictionary
def mood_from_song_features(song_features):
    valence = song_features.get('valence', 0.5)
    energy = song_features.get('energy', 0.5)
    return classify_mood(valence, energy)

# Global variables for ML models -
spotify_df = None
comparator = None
best_model = None
best_name = None

def initialize_ml_models():
    global spotify_df, comparator, best_model, best_name
    if comparator is None:
        spotify_df = load_Spotify_data()
        comparator = Comparison(spotify_df)
        comparator.train_models_evaluate()
        best_name, best_model = comparator.get_best_model()

def mood_from_song_features_ml_models(song_features):
    if comparator is None:
        initialize_ml_models()
    
    features_df = pd.DataFrame([song_features])
    features_scaled = comparator.scaler.transform(features_df)

    mood_numeric = best_model.predict(features_scaled)[0]
    mood_mapping_reverse = {0: 'Happy', 1: 'Sad', 2: 'Energetic', 3: 'Calm'}
    return mood_mapping_reverse[mood_numeric]

# Example usage
if __name__ == "__main__":
    test_songs = [
        {'name': 'Happy Song', 'valence': 0.8, 'energy': 0.7},
        {'name': 'Sad Song', 'valence': 0.2, 'energy': 0.3},
        {'name': 'Energetic Song', 'valence': 0.9, 'energy': 0.8},
        {'name': 'Calm Song', 'valence': 0.4, 'energy': 0.5}
    ]

    for song in test_songs:
        mood = mood_from_song_features(song)
        print(f"{song['name']}: Mood is {mood}")

   