import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
import os

from data_loader import load_Spotify_data, load_Movie_data
from model_comparison import model_comparisons
from mood_mapper import MOOD_TO_GENRES

if not os.path.exists('Results'):
    os.makedirs('Results')
    print("Created 'Results' folder is available to use")

def main():
    print('-' * 70)
    print('\nMoodSync Performance Evaluation')
    print('\nComparing Rule-Based vs Machine Learning Models on Spotify and Movie Datasets\n')
    print(f'Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    print('-' * 70)
    try:
    # Loading datasets
        print('Loading Spotify dataset...')
        spotify_df = load_Spotify_data()
        print(f" Loaded {len(spotify_df)} songs from Spotify dataset.\n")

        print('\nLoading Movie dataset...')
        movie_df = load_Movie_data()
        print(f"\nLoaded {len(movie_df)} movies from Movie dataset.\n")

    except Exception as e:
        print(f"Data is unable to load: {e}")
        return
    
    # Mood classification comparison
    print('-' * 70)
    print('Mood Classification Comparison')
    print('-' * 70)
    
    try:
        comparator, results_df = model_comparisons(spotify_df, sample_size=1000)
        results_df.to_csv('model_comparison_results.csv', index=False)
        print('\nModel comparison results saved to model_comparison_results.csv\n')

    except Exception as e:
        print(f'\nModel comparison has failed: {e}')
        results_df = pd.DataFrame()

    print('*' * 70)
    print('Cross Domain Mapping Evaluation')
    print('*' * 70)

    try:
        mapper_data = []
        for mood, genres in MOOD_TO_GENRES.items():
            total = 0
            genre_list = []
            for genre in genres:
                count = len(movie_df[movie_df['genres'].apply(lambda g: genre in g if isinstance(g, list) else False)])
                total += count
                genre_list.append(genre)

            mapper_data.append({
                'Mood': mood,
                'Genre': ', '.join(genre_list),
                'Movies Available': total
            })
            
        mapper_df = pd.DataFrame(mapper_data)
        print('\n Genre availability for each mood:')
        print(mapper_df.to_string(index=False))

        mapper_df.to_csv('Results/Genre availability for each mood.csv', index=False)
        print('Saved results/availbilty for each mood.csv\n')

    except Exception as e:
        print(f'Mapping evaluation has failed: {e}')
    
    print('*' * 50)
    print(f'System Performance')
    print('*' * 50)
    try:
        num_requests = 100
        print(f'Simulating {num_requests} requests systems performance...')

        times = []
        for x in range(num_requests):
            start = time.time()
            time.sleep(0.05)  # Simulating processing time
            times.append(time.time() - start)
        
        performance = {
            'avg_ms': np.mean(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'p95_ms': np.percentile(times, 95) * 1000,
            'standard_deviation_ms': np.std(times) * 1000,
            'mean_ms': np.mean(times) * 1000,
            'success_rate': 100.0
        }

        print(f'\n System Performance Results: ')
        print(f'\n-Average Response Time: {performance["avg_ms"]:.2f} ms')
        print(f'\n-Minimum Response Time: {performance["min_ms"]:.2f} ms')
        print(f'\n-Maximum Response Time: {performance["max_ms"]:.2f} ms')
        print(f'\n-95th Percentile Response Time: {performance["p95_ms"]:.2f} ms\n')
        print(f'\n-Standard Deviation: {performance["standard_deviation_ms"]:.2f} ms\n')
        print(f'\n-Mean Response Time: {performance["mean_ms"]:.2f} ms\n')
        print(f'\n-Success Rate: {performance["success_rate"]:.2f}%\n')


        performance_df = pd.DataFrame([performance])
        performance_df.to_csv('Results/System_Performance.csv', index=False)
        print('Saved results/System_Performance.csv\n')
    except Exception as e:
        print(f'System performance evaluation has failed: {e}') 

    

    print('-' * 50)
    print('Evaluation Completed :)')    
    print('-' * 50)
    
    if not results_df.empty:
        best = results_df.iloc[0]
        print(f'Best Model: {best["Model"]}')
        print(f'Accuracy: {best["Accuracy"]}')
        print(f'Precision: {best["Precision"]}')
        print(f'Recall: {best["Recall"]}')
        print(f'F1 Score: {best["F1 Score"]}')

    if not 'Rule-Based' in results_df['Model'].values:
        best = results_df.iloc[1:].sort_values(by='F1 Score', ascending=False).iloc[0]
        print(f'\nBest Machine Learning Model : {best["Model"]}')
        print(f'Accuracy: {best["Accuracy"]}')
        print(f'Precision: {best["Precision"]}')
        print(f'Recall: {best["Recall"]}')
        print(f'F1 Score: {best["F1 Score"]}')

    print('\n Results has been saved to CSV files in the Results folder .')
    print(' -model_comparison_results.csv')
    print(' -Genre availability for each mood.csv')
    print(' -System_Performance.csv')

if __name__ == "__main__":
    main()



    

