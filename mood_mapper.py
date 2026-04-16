# Mood to Movie Genres Mapping
MOOD_TO_GENRES = {
    'Happy': ['Comedy', 'Animation', 'Family', 'Musical', 'Adventure'],
    'Sad': ['Drama', 'Romance', 'History', 'War'],
    'Energetic': ['Action', 'Thriller', 'Horror', 'Sci-Fi'],
    'Calm': ['Documentary', 'Biography', 'Mystery', 'Fantasy', 'Western', 'TV Movie']
}

# Function to get genres for a given mood
def get_genres_for_mood(mood):
    return MOOD_TO_GENRES.get(mood, [])

# Function to recommend movie genres based on mood
def recommend_genres_for_mood(mood, movies_df, top_n=5):
    target_genres = get_genres_for_mood(mood) # Get genres for the specified mood
    # Filter movies that match the target genres
    # Create a boolean mask for movies matching the target genres
    mask = movies_df['genres'].apply(lambda genres: any(g in target_genres for g in genres))

    # Filter and sort movies based on ratings
    recommendations = movies_df[mask].copy()
    recommendations = recommendations.sort_values(by='vote_average', ascending=False).head(top_n)

    return recommendations.head(top_n)

# Test example
if __name__ == "__main__":
    import pandas as pd 

    # Sample movie data
    sample_movies = pd.DataFrame({
        'title' : ['ToyStory', 'The Dark Knight', 'Titantic', 'Finding Nemo', 'The Shawshank Redemption'],
        'genres' : [['Animation',  'Comedy', 'Family'],
                    ['Action', 'Crime', 'Drama'],
                    ['Drama', 'Romance'],
                    ['Animation', 'Adventure', 'Comedy', 'Family'],
                    ['Drama']],
                'vote_average' : [7.7, 9.0, 7.8, 8.1, 9.3]
                    
    })


    # Test recommendations
    moods = ['Happy', 'Sad', 'Energetic', 'Calm']
    for mood in moods:
        print(f"\n  Movies for {mood} mood:")
        recs = recommend_genres_for_mood(mood, sample_movies, top_n=2)
        for _, row in recs.iterrows():
            print(f" - {row['title']} (Rating: {row['vote_average']}/ 10)")

