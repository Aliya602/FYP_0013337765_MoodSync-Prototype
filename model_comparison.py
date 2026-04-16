import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from datetime import datetime
import time
import json
from sklearn.preprocessing import StandardScaler

# ML models for mood classifcation vs rule based baseline
class Comparison:
    def __init__(self, songs_df):
        self.songs_df = songs_df.copy()
        self.results = {}
        self.scaler = StandardScaler()

    def rule_based_classification(self):
        # A simple rule-based classification based on valence and energy as a baseline for comparison with ML models.
        print('Applying Rule-Based Classification (Baseline)....')
        
        def rule_based_mood(row):
            valence = row['valence']
            energy = row['energy']
            if valence > 0.65 and energy > 0.55:
                return 'Happy'
            elif valence < 0.35 and energy < 0.35:
                return 'Sad'
            elif energy > 0.70:
                return 'Energetic'
            else:
                return 'Calm'
        
        self.songs_df['rule_based_mood'] = self.songs_df.apply(rule_based_mood, axis=1)

        print('\nDistribution:')

        for mood, count in self.songs_df['rule_based_mood'].value_counts().items():
            print(f'{mood}: {count} songs ({count/len(self.songs_df)*100:.2f}%)')
        
        return self.songs_df
    
    def prepare_features_and_lables(self):
        if 'rule_based_mood' not in self.songs_df.columns:
            self.rule_based_classification()

        # Defining the features to be used for ML models.The same features as the rule-based model for a fair comparison.
        feature_cols = ['valence', 'energy', 'tempo', 'danceability', 'loudness']
        available = [col for col in feature_cols if col in self.songs_df.columns]

        X = self.songs_df[available].copy()
        X = X.fillna(X.mean())
        X_scaled = self.scaler.fit_transform(X)

        mood_mapping = {'Happy': 0, 'Sad': 1, 'Energetic': 2, 'Calm': 3}
        self.songs_df['rule_based_mood_numeric'] = self.songs_df['rule_based_mood'].map(mood_mapping)
        y = self.songs_df['rule_based_mood_numeric'].values

        return X_scaled, y, available

        
    
    def train_models_evaluate(self, test_size=0.2, random_state=42):
        # Training multiple ML models and evaluating their performance against the rule-based baseline.
        
        print('Training ML Models and Evaluating Performance...\n')
        
        X, y, feature_names = self.prepare_features_and_lables() 
        indices = np.arange(len(X))

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, test_size=test_size, random_state=random_state, stratify=y)
        
        print('\nDataset info:')
        print(f'Total Samples: {len(X)}')
        print(f'Features Used: {feature_names}')
        print(f'Training Samples: {len(X_train)})')
        print(f'Testing Samples: {len(X_test)}\n') 
 

        self.evaluate_rule_baseline(idx_test,y_test)

        models ={
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='rbf', random_state=42)
        }

        print('*'*70)
        print('Training and Evaluating Models...')
        print('*'*70)

        for name, model in models.items():
            print(f'Training {name}...')
            start = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            self.results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'training_time': time.time() - start
            }

            print(f'\nResults for {name}:')
            print(f"Accuracy: {self.results[name]['accuracy']:.4f}")
            print(f"Precision: {self.results[name]['precision']:.4f}")
            print(f"Recall: {self.results[name]['recall']:.4f}")
            print(f"F1 Score: {self.results[name]['f1_score']:.4f}")
            print(f"Training Time: {self.results[name]['training_time']:.2f}\n")

        return self.results
    
    def evaluate_rule_baseline(self, idx_test, y_test ):
        # Evaluating the performance of the rule-based baseline for comparison with ML models.
        
        print('Evaluating Rule-Based Baseline...')
        

        test_positions  = np.array(idx_test).flatten().tolist()

        rule_based_predictions = []
        for pos in test_positions:
            pos = int(pos)
            valence = self.songs_df.iloc[pos] ['valence']
            energy = self.songs_df.iloc[pos] ['energy']
            if valence > 0.65 and energy > 0.55:
                rule_based_predictions.append(0)  # Happy mood
            elif valence < 0.35 and energy < 0.35:
                rule_based_predictions.append(1)  # Sad mood
            elif energy > 0.70:
                rule_based_predictions.append(2)  # Energetic mood
            else:
                rule_based_predictions.append(3)  # Calm mood

        y_pred_rule= np.array(rule_based_predictions)

        self.results['Rule-Based Baseline'] = {
        'model': None,
        'accuracy': accuracy_score(y_test, y_pred_rule),
        'precision': precision_score(y_test, y_pred_rule, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_rule, average='weighted'),
        'f1_score': f1_score(y_test, y_pred_rule, average='weighted'),
        'training_time': 0.0
        }
        print('Results for Rule-Based Baseline:')
        print(f'Accuracy: {self.results["Rule-Based Baseline"]["accuracy"]:.4f}')
        print(f'Precsion: {self.results["Rule-Based Baseline"]["precision"]:.4f}')
        print(f'Recall: {self.results["Rule-Based Baseline"]["recall"]:.4f}')
        print(f'F1 Score: {self.results["Rule-Based Baseline"]["f1_score"]:.4f}')

    def recommendation_songs_to_movies(self):
        # A simple recommendation of songs to movies based on the predicted mood from the best performing model.
        best_name, best_model = self.get_best_model()
        if best_model is None:
            print('No ML models were trained, only the rule-based baseline is available. Cannot make recommendations based on ML predictions.')
            return None
        
        print(f'\nMaking song recommendations based on the best performing model: {best_name}...\n')

        X, _, _ = self.prepare_features_and_lables()
        predicted_moods = best_model.predict(X)

        mood_mapping_reverse = {0: 'Happy', 1: 'Sad', 2: 'Energetic', 3: 'Calm'}
        self.songs_df['predicted_mood'] = [mood_mapping_reverse.get(pred, 'Unknown') for pred in predicted_moods]

        recommendations = self.songs_df.groupby('predicted_mood').apply(lambda x: x.sample(5, random_state=42)).reset_index(drop=True)
        
        print('Recommended Songs for Each Predicted Mood:')
        for mood in recommendations['predicted_mood'].unique():
            print(f'\nMood: {mood}')
            mood_songs = recommendations[recommendations['predicted_mood'] == mood]
            for idx, row in mood_songs.iterrows():
                print(f"- {row['name']} by {row['artist']} (Valence: {row['valence']:.2f}, Energy: {row['energy']:.2f})")

        return recommendations

    def results_summary(self):
        # Summarizing and displaying the results
        if not self.results:
            print('There are no results to summarize yet. Please run train_models_evaluate() function first.')
            return None
        
        print('-'* 60)
        print('Model Comparison Summary:')
        print('-'* 60)

        rows = []
        for name, metrics in self.results.items():
            rows.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}",
                'Training Time (sec):': f"{metrics['training_time']:.2f}"
            })

        summary_df = pd.DataFrame(rows)
        summary_df = summary_df.sort_values('F1 Score', ascending=False)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def get_best_model(self):
        # Returns the best performing model name and model
        ml_results = {k: v for k, v in self.results.items() if k != 'Rule-Based Baseline'}
        if ml_results:
            best = max(ml_results.items(), key=lambda x: x[1]['f1_score'])
            return best [0], best[1]['model']
        
        return None,None
    
    def report(self):
        # Generating a report of the results in JSON format for further analyisis
        report = {
            'timestamp': datetime.now().isoformat(),
            'results' : {}
        }
        for name, metrics in self.results .items():
            report['results'][name] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'training_time': metrics['training_time']
            }
        with open('model_comparison_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        print(f' Report saved to model_comparison_report.json ')
        return report
    


def model_comparisons(songs_df, sample_size=1000):
    # Run model comparison on a sample dataset for quick evaluation
    print('*'* 60)
    print('\nStarting Model Comparison...')
    print('Comparing Rule-Based Baseline with ML Models...\n')

    if len(songs_df) > sample_size:
        print(f'Sampling {sample_size} songs from the dataset')
        songs_subset = songs_df.sample(sample_size, random_state=42).reset_index(drop=True)
    else:
        songs_subset = songs_df.reset_index(drop=True)

    comparator = Comparison(songs_subset)
    comparator.train_models_evaluate()
    summary_df = comparator.results_summary()
    comparator.report()

    best_name, best_model = comparator.get_best_model()
    if best_name:
        print(f'\nBest Performing Model: {best_name}')
        print(f'Best Model Details: {best_model}')
        print(f'F1 Score: {comparator.results[best_name]["f1_score"]:.4f}')
    else:
        print('No ML models were trained, only the rule-based baseline is available.')

    return comparator,summary_df    


if __name__ == '__main__':
    print('Running Model Comparison...')
    print('from data_loader.model_comparison import model_comparison')
    print('df = load_Spotify_data()')
    print('run model_comparison(df)')

    
        


    


        



    
  



    

