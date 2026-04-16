from flask import Flask, request, jsonify #creats app, recieve data, sends data back
from flask_cors import CORS
import pandas as pd

# Getting functions from previous files
from data_loader import load_Movie_data , load_Spotify_data 
from mood_classifier import mood_from_song_features
from mood_mapper import recommend_genres_for_mood

app = Flask(__name__)  #create object from web server

CORS(app)  # Streamlit can communicate with Flask

spotify_df = None
movies_df = None


# loading the data
def load_data():
    global spotify_df, movies_df
    print("Loading Datasets...")

    # Call the loader functions to get DataFrame objects
    spotify_df = load_Spotify_data()
    print(f" Spotify: {len(spotify_df)} songs loaded")
    movies_df = load_Movie_data()
    print(f" Movies: {len(movies_df)} songs loaded")

    print("Datasets loaded successfully...")

load_data() #runs the function
    


# Homepage Endpoint
@app.route("/")
def home(): # runs when someone visits '/'
    return jsonify({  # converts the python dictionary to JSON format
        'message': 'MoodSync is running!',
        'endpoints': {
            '/': 'GET - This help message',
            '/test': 'GET - Check if API is working',
            '/songs': 'GET - List available songs',
            '/search_songs': 'GET - Search songs by name (add ?q=songname)',
            '/recommend': 'POST - Get film recommendations for a song'
        }
    })


# Here it verifies whether the API is working (Test Endpoint)
@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status' : 'success',
        'message' : 'MoodSync API is running...',
        'data_loaded' : {
            'songs': len(spotify_df) if spotify_df is not None else 0, # checks if data exists first
            'movies': len(movies_df) if movies_df is not None else 0

        }
    })


# get first 100 songs
@app.route('/songs', methods =['GET'])
def get_songs():
    try:
        songs = []
        for idx, row in spotify_df.head(100).iterrows():
            songs.append({
                'id': row['id'],
                'name': row['name'],
                'artists': row['artists'],
                'release_date': row['release_date']
            })
        return jsonify({'count': len(songs), 'songs': songs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Search Songs Endpoint (Feature)
@app.route('/search_songs', methods=['GET'])
def search_songs():
    try:
        # Get what user typed
        query = request.args.get('q', '').lower().strip() #Get search query from URL parameters
        decade = request.args.get('decade', 'all')
        
        if not query or len(query) < 2:
            return jsonify({
                'songs': [],
                'count': 0,
                'message': 'Please enter atleast 2 or 3 characters'
            })
        
        filtered_df = spotify_df.copy()

        filtered_df['year'] = pd.to_datetime(filtered_df['release_date'], errors='coerce').dt.year

        #decade filter
        #Keeps only songs from 2000 or later
        if decade != 'all':
            if decade == '2000s':
                filtered_df = filtered_df[(filtered_df['year'] >= 2000) & (filtered_df['year'] < 2010)]
            elif decade == '2010s':
                filtered_df = filtered_df[(filtered_df['year'] >= 2010) & (filtered_df['year'] < 2020)]
            elif decade == '2020s':
                filtered_df = filtered_df[(filtered_df['year'] >= 2020)]
        
        # Search song names (case-insensitive)
        # Apply mask variable to get matching songs
        mask = filtered_df['name'].str.lower().str.contains(query, na=False)
        results = filtered_df[mask].head(20)  # Limit to 20 results for performance

        # Results for JSON response
        songs = []
        for idx , row in results.iterrows():
            songs.append({
                'id': row['id'],
                'name': row['name'],
                'artists': row['artists'],
                'year': int(row['year']) if pd.notna(row['year']) else 'Unknown',
                'valence': row['valence'],
                'energy': row['energy']
            })

        # Return results
        return jsonify({
            'query': query,
            'decade': decade,
            'count': len(songs),
            'songs': songs
        })
    
    # return error incase anything goes wrong
    except Exception as e:
        return jsonify({'error': str(e), 'songs': []}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json

        # Extract song_id from the data
        song_id = data.get('song_id')
        
        # If song_id doesn't exists, it returns an error message
        if not song_id:
            return jsonify({'error': 'No song_id provided'}), 400
        
        # Find the song from the spotify dataset
        song = spotify_df[spotify_df['id']== song_id]  

        # Check if the song exists/ if not, it shows an error
        if song.empty:
            return jsonify({'error': 'Song not found'}), 404
        
        song = song.iloc[0] # Getting the first (and only) matching song
        
        # Extracting features
        song_features = {
            'valence': song['valence'],
            'energy': song['energy']
        }
        
        # Calls your function which returns 'Happy', 'Sad', 'Energetic', or 'Calm'
        mood = mood_from_song_features(song_features)
        
        # Get film recommendations from mood
        recommendations = recommend_genres_for_mood(mood, movies_df, top_n=5)
        
        # Response format
        response = {
            'song' : {
                'id' : song['id'],
                'name': song['name'],
                'artists': song['artists'],
                'valence': float(song['valence']),
                'energy': float(song['energy']),
                'mood': mood
            },

            'recommendations' : recommendations[['title', 'genres', 'vote_average', 'vote_count']].to_dict('records')
            }

        return jsonify(response)
    
    # If anything crashes, there will be an error
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        

        
if __name__ == '__main__':
    print("\n" + "="*50)
    print(" Starting MoodSync API server...")
    print(f" API will be available at: http://127.0.0.1:5000")
    print("="*50)
    print("\n Available endpoints:")
    print("   • GET  / - API information")
    print("   • GET  /test - Check API status")
    print("   • GET  /songs - List first 100 songs")
    print("   • GET  /search_songs?q=love&decade=2020s - Search songs")
    print("   • POST /recommend - Get film recommendations")
    print("\n" + "="*50)
    
    # Run the Flask app
    app.run(debug=True, port=5000)

