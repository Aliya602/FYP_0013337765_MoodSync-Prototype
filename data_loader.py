import pandas as pd
import ast


# Loading Spotify dataset
def load_Spotify_data():
    df = pd.read_csv('data_loader/spotify_overview.csv')
    base_cols = ['id', 'name', 'artists', 'release_date','danceability', 'energy', 'loudness', 'valence', 'tempo']
    
    optional_cols = ['acousticness', 'liveness', 'speechiness', 'instrumentalness', 'key' ]

    available_cols = base_cols.copy()
    for i in optional_cols:
        if i in df.columns:
            available_cols.append(i)
    df = df[available_cols]
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

    print(f'Loaded Songs: {len(df)} songs')
    print(f'\nFeatures Available: {available_cols}')
    print(f'\nSample songs:\n{df[["name", "artists", "valence", "energy"]].head()}')
    
    return df


#Loading Movie Dataset
def load_Movie_data(filepath='data_loader/movies_metadata.csv'):
    df = pd.read_csv(filepath, low_memory=False)
    df = df[['id', 'title', 'genres', 'release_date', 'popularity', 'vote_average', 'vote_count']]

    def parse_genres(genre_str):
        try:
            # Convert string representation of list of dictionaries to actual list of genre names
            if pd.isna(genre_str):
                return []
            genres = ast.literal_eval(genre_str)
            return [g['name'] for g in genres]
        except:
            return []
        
    # Parsing genres from string representation of list of dictionaries
    df['genres'] = df['genres'].apply(parse_genres)
    df = df[df['vote_count']>50]
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    
    print('*'*70)
    print(f'Loaded {len(df)} movies')
    print(f'\nColumns: {list(df.columns)}')
    print(f'\nSample Genres: {df["genres"].explode().value_counts().head(10).to_dict()}')
    
    
    return df

def check_data(df, name='Dataset'):
    print(f'{name} Quality Check ->')
    print(f'\nTotal rows: {len(df)}')
    print(f'\nColumns: {list(df.columns)}')
    print('*'*70)

    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    if len(missing_cols) > 0:
        print(f'Missing values: {dict(missing_cols)}')
    else:
        print(f'\nNo Missing Values...')
    if 'id' in df.columns:
        duplicates = df['id'].duplicated().sum()
        print(f'\nDuplicate ID: {duplicates}')

    print(f'\nData types: {df.dtypes.value_counts().to_dict()}')
    
    return missing_values.sum()

def sample_data(df, n=10000, random_state=42):
    # Take random samples for faster evaluation
    if len(df) > n:
        return df.sample(n=n, random_state=random_state)
    return df
   

def get_available_features(df):
    # audio features that are available for ML 
    features_cols = ['valence', 'energy', 'tempo', 'danceability', 'loudness', 'Acousticness', 'Liveness', 'Speechiness', 'Instrumentalness']
    available = [col for col in features_cols if col in df.columns]
    missing = [col for col in features_cols if col not in df.columns]

    print(f'Available audio features for ML: {available}')
    if missing:
        print(f'Missing Features: {missing}')

    return available    


if __name__ == "__main__":
    # Main Test
    print('*'*70)
    print('Data Loader Test...')
    print('*'*70)

    spotify_data = load_Spotify_data()
    movie_data = load_Movie_data()

    check_data(spotify_data, 'Spotify Dataset')
    print('\n' + '*'*70)
    check_data(movie_data, 'Movie Dataset')

    get_available_features(spotify_data)
    print('\n' + '*'*70)
    print('Spotify Data Sample(first 5 rows):')
    print(spotify_data[['name', 'artists', 'valence', 'energy', 'tempo']].head())

    print('\n' + '*'*70)
    print('Movie Data Sample(first 5 rows):')
    print(movie_data[['title', 'genres', 'vote_average']].head())

    print('\n Data loader test successfully completed!')
