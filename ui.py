import streamlit as st
import pandas as pd
import requests
import time
import json
import random as rdom
from datetime import datetime

st.set_page_config(
    page_title='MoodSync',
    page_icon='🎵',
    layout='wide'
)

#This is where the Flask backend is running
API = 'http://127.0.0.1:5000' 

#CSS for styling
st.markdown("""
<style>
.main-header { font-size: 3rem; color: #1DB954; text-align: center; margin-bottom: 1rem; }
.sub-header { text-align: center; color: #3266a8; margin-bottom: 2rem; }
.song-card { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 5px solid #1DB954; cursor: pointer; transition: transform 0.2s; }
.song-card:hover { transform: translateX(5px); background-color: #e8eaf0; }
.movie-card { background-color: #e8f4f8; padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #1DB954; }
.mood-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: bold; margin: 0.5rem 0; }
.happy { background-color: #FFD700; color: #000; }
.sad { background-color: #4682B4; color: white; }
.energetic { background-color: #4682B4; color: white; }
.calm { background-color: #32CD32; color: white; }
.error-box { background-color: #ffebee; color: #c62828; padding: 1rem; border-radius: 10px; border-left: 5px solid #c62828; }
</style>
""", unsafe_allow_html=True)


#page header
st.markdown('<h1 class="main-header">🎵 MoodSync 🎬</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover films that match your music mood</p>', unsafe_allow_html=True)

#Intialize session state - remembers thngs between interactions
# serves as memory
if 'search_results' not in st.session_state:
    st.session_state.search_results = [] #stores songs from search

if 'selected_song' not in st.session_state:
    st.session_state.selected_song = None

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False

# Create a sidebar
with st.sidebar:
    st.header("About MoodSync")
    # How the API works
    st.markdown("""
    This system explores whether the mood of muis can be mapped to film recommendations.    
    1.Search for a song (1920-2020)
    2.Select from matching results
    3.Get mood-matched film recommendation
    """)

    st.divider()

    st.header("API Status")
    try:
        response = requests.get(f"{API}/test", timeout=3)
        if response.status_code == 200: # Connecting to Flask
            data = response.json()
            st.success("Connected")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Songs", f"{data["data_loaded"]["songs"]:,}")
            with col2:
                st.metric("Movies", f"{data["data_loaded"]["movies"]:,}")
        else:
            st.error("Error with the API")
    except Exception as e:
        st.error("Unable to connect to the API....")
        st.info("Make sure Flask is runnng")
    st.divider()
    st.caption("2024 MoodSync - FYP")

col1, col2 = st.columns([1, 1.5]) # Main content

# Search Section - left columns
with col1:
    st.header("Search Songs")
    search_query = st.text_input(
        "Song name: ",
        placeholder="e.g., 'Blinding Lights'",
        key="search_input"
    )
   #Filtering Decade
    decade = st.selectbox(
            "Filter by decade:",
            options=['all', '2000s', '2010s', '2020s' ],
            index=2, #defult is 2020s
            help="Focus on songs from specific decades" 
        )

        # Search and Clear buttons side by side
    search_col1, search_col2 = st.columns([1,1])
    with search_col1:
        search_button = st.button("Search", type="primary", use_container_width=True)
    with search_col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)

    if clear_button:
            st.session_state.search_results = []
            st.session_state.selected_song = None
            st.session_state.recommendations = None
            st.session_state.search_performed = False
            st.rerun() #R efresh the page when user presses clear- reset all session state

        # Handle search button
    if search_button and search_query:
            with st.spinner(f"Searching for '{search_query}'..."):
                try:
                    response = requests.get(
                    f"{API}/search_songs",
                    params={"q": search_query, "decade": decade}
                )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.search_results = data['songs']
                        st.session_state.search_performed = True

                        if data['count'] == 0:
                            st.warning(f"No songs matching '{search_query}'")
                        else:
                            st.success(f"Found {data['count']} songs")
                except Exception as e:
                    st.error(f"Error:{e}")
        
    if st.session_state.search_results:
            st.divider()
            st.markdown(f"Results ({len(st.session_state.search_results)})")

            for song in st.session_state.search_results:
                card_key = f"song_{song['id']}"
                is_selected = (st.session_state.selected_song and st.session_state.selected_song['id'] == song['id'])
                
                if is_selected:
                    st.markdown(f"""
                    <div class="song-card"  style="border-left: 5px solid  #1DB954; background-color: #e1f5e1;">
                    <strong>{song['name']}</strong><br>
                    <span style='color:#666;'>{song['artists']}</span></br>
                    <span style='font-size: 0.8rem;'>Year: {song.get('year','Unknown')} | Valence: {song['valence']:.2f}, Energy: {song['energy']:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("✅ Selected - Click 'Get Recommendations'")
                else:
                    if st.button(f"🎵 {song['name']} - {song['artists'][:50]}...", key=card_key, use_container_width=True):
                        st.session_state.selected_song = song
                        st.session_state.recommendations = None
                        st.rerun()

    elif st.session_state.search_performed and not st.session_state.search_results:
        st.markdown("""
        <div class = "error-box">
            <strong> 🔎 Song not Found</strong><br>
            Try:
            <ul>
                <li>Checking spelling</li>
                <li>Using fewer words</li>
                <li>Selecting a different decade</li>
                <li>Trying another song</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Showing recommendation button if a song is selected
    if st.session_state.selected_song:
        st.divider()
        st.markdown('Get Recommendations')

        if st.button("Get Film Recommendations", type="primary", use_container_width=True):
                        with st.spinner('Analyzing mood and finding films...'):
                            try:
                                response = requests.post(f"{API}/recommend", json={'song_id': st.session_state.selected_song['id']}, timeout=10)
                                if response.status_code == 200:
                                    st.session_state.recommendations = response.json()
                                    st.success('Recommendation ready ;)!')
                                    st.balloons()
                                else:
                                    st.error('Failed to get recommendation')
                            except Exception as e:
                                st.error(f"Error: {e}")

# Recommendation - Right column
with col2:
    st.header('Recommendations')
    if st.session_state.recommendations:
        data = st.session_state.recommendations

# Selected song info
        st.markdown(f"""
        <div class="song-card">
            <h4> Selected: {data['song']['name']}</h4>
            <p>Artists: {data['song']['artists']}</p>
            <p>Valence: {data['song']['valence']:.3f} | Energy: {data['song']['energy']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

# Displaying the mood
        mood = data['song']['mood']
        mood_colours = {'Happy': 'happy', 'Sad': 'sad', 'Energetic': 'energetic', 'Calm': 'calm'}

        st.markdown(f"""
        <div style='text-align: center; margin: 1rem 0;'>
                    <h3>Detected Mood: </h3>
                    <span class="mood-badge {mood_colours.get(mood, '')}" style="font-size:1.2rem;">
                    {mood}
                    </span>
        </div>
         """ , unsafe_allow_html=True)

         # Movie Recommendation
        if data['recommendations']:
            st.subheader(f"Top Films for {mood} Mood")

            # Handle genres that might be a list or string
            for i, movie in enumerate(data['recommendations'], 1):
                if isinstance(movie['genres'], list):
                    genres = movie['genres']
                else:
                    try:
                        genres = json.loads(movie['genres'])
                    except:
                        genres = [movie['genres']]
                
                st.markdown(f"""
                <div class='movie-card'>
                    <h4>{i}. {movie['title']}</h4>
                    <p><strong>Genres:</strong> {', '.join(str(g) for g in genres[:5])}{'...' if len(genres) > 5 else ''}</p>
                    <p><strong>Rating:</strong> ⭐ {movie['vote_average']:.1f}/10 ({movie['vote_count']:,} votes)</p>
                    <img src="{movie.get('poster_path', '')}" alt='Poster' style='float:right; border-radius:5px; margin-left:10px;' onerror="this.onerror=null; this.src='https://via.placeholder.com/100x150?text=No+Image';">
                </div>
                """, unsafe_allow_html=True)

            df = pd.DataFrame(data['recommendations'])
            csv = df[['title', 'vote_average']].to_csv(index=False)

            st.download_button(
                label="📤 Download Recommendations",
                data=csv,
                file_name=f"moodsync_{mood}_{data['song']['name'][:20]}.csv", mime="text/csv"
            )
        else:
                st.warning(" Sorry, there is no films found for this mood. Try another song!")
    elif st.session_state.selected_song:
            st.info("🔎 Click 'Get Film Recommendations' to see mood-matched films")
    else:
        st.info("🔎 Search for song to get started..")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎶 <strong>MoodSync</strong> - Music → Mood → Movies | Songs: 1920-2020</p>
        
    </div>
    """, unsafe_allow_html=True)
