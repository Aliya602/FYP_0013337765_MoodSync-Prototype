import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

plt.style.use('ggplot')
sns.set_style('whitegrid')



# Model performance data visualization
models = ['Rule-Based', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'SVM', 'KNN']
f1_scores_results = [1.000, 0.9900, 0.9900, 0.9199, 0.9046, 0.8997]
accuracy_results = [1.000, 0.9900, 0.9900, 0.9200, 0.9050, 0.9000]

fig, ax = plt.subplots(figsize = (10,6))
x = np.arange(len(models))
width = 0.35
bars_1 = ax.bar(x-width/2, f1_scores_results, width, label='F1 Score', color = '#4D497A', edgecolor='black', linewidth=0.5) 
bars_2 = ax.bar(x+width/2, accuracy_results, width, label='Accuracy', color = "#2AA49A", edgecolor='black', linewidth=0.5) 

ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_ylabel('Score')
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
ax.legend(loc= 'lower right')
ax.set_ylim(0.85, 1.02)

for bars in bars_1:
    height = bars.get_height()
    ax.annotate(f'{height:.4f}', xy=(bars.get_x() + bars.get_width()/2, height), xytext=(0,3), textcoords='offset points', ha='center', va='bottom')

for bars in bars_2:
    height = bars.get_height()
    ax.annotate(f'{height:.4f}', xy=(bars.get_x() + bars.get_width()/2, height), xytext=(0,3), textcoords='offset points', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('Results/performance_comparison_chart.png', dpi=300, bbox_inches='tight')
print('Generating performance comparison chart...')
plt.show()
print('Performance comparison chart saved to Results/performance_comparison_chart.png')


# genre availability for each mood visualization
def plot_genre_availability(mapper_df):
    moods = mapper_df['Mood']
    movie_availability = mapper_df['Movie Genres Available']
    colors = ['#4D497A', "#E78FDF", "#E5E88E", '#FF6F61']

    fig, ax = plt.subplots(figsize=(10,6))
    bar = ax.bar(moods, movie_availability, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title('Genre Availability for Each Mood', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Movie Genres Available')

    for x in bar:
        height = x.get_height()
        ax.annotate(f'{height}', xy=(x.get_x() + x.get_width()/2, height), xytext=(0,3), textcoords='offset points', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('Results/genre_availability_chart.png', dpi=300, bbox_inches='tight')
    print('Generating genre availability chart...') 
    plt.show()
    print('Genre availability chart saved to Results/genre_availability_chart.png')

moods = ['Happy', 'Sad', 'Energetic', 'Calm']
movie_genres_available = [6186, 6267, 5805, 1989]
Genre_available = [
    ['Comedy', 'Animation', 'Family', 'Musical', 'Adventure'],
    ['Drama', 'Romance', 'War', 'History', 'Thriller'],
    ['Action', 'Sci-Fi', 'Fantasy', 'Horror', 'Mystery'],
    ['Documentary', 'Biography', 'Music', 'Sport', 'Western']
]
mapper_df = pd.DataFrame({
    'Mood': moods,
    'Movie Genres Available': movie_genres_available,
    'Genre': Genre_available 
})   
plot_genre_availability(mapper_df)

# Accuracy vs Training Time visualization
models = ['Rule-Based', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'SVM', 'KNN']
accuracy_results = [1.000, 0.9900, 0.9900, 0.9200, 0.9050, 0.9000]
times = [0.00, 0.01, 0.15, 0.50, 0.02, 0.01]

colors = ['#4D497A', '#2AA49A', '#E94F37', '#F6C90E', '#1B998B', '#FF6F61']

fig, ax1 = plt.subplots(figsize=(10,6))

ax1.bar(models, accuracy_results, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Models')
ax1.set_ylabel('Accuracy', color='black')   
ax1.set_title('Model Accuracy vs Training Time', fontsize=14, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='black')
ax2 = ax1.twinx()
ax2.plot(models, times, color='black', marker='o', linestyle='--', label='Training Time (sec)')
ax2.set_ylabel('Training Time (sec)', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.legend(loc='upper left')

for i, v in enumerate(accuracy_results):
    ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
for i, v in enumerate(times):
    ax2.text(i, v + 0.01, f'{v:.2f} sec', ha='center', va='bottom') 

plt.tight_layout()
plt.savefig('Results/accuracy_vs_training_time_chart.png', dpi=300, bbox_inches='tight')
print('Generating accuracy vs training time chart...')  
plt.show()
print('Accuracy vs Training Time chart saved to Results/accuracy_vs_training_time_chart.png')




#accuracy comparison visualization line chart
def plot_accuracy_comparison(models, accuracy_results):
    models = models
    accuracy_results = accuracy_results
    colors = [plt.cm.Blues(np.linspace(0.2, 0.8, len(models)))]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(models, accuracy_results, 'o-', color="#DF4C62", markerfacecolor='white', markeredgecolor='black', linewidth=2, markersize=8)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison of ML Models', fontsize=14, fontweight='bold')
    ax.set_ylim(0.85, 1.02)
    ax.grid(True, alpha=0.3)

    for i, (model, v) in enumerate(zip(models, accuracy_results)):
        ax.annotate(f'{v:.4f}', xy=(i, v), xytext=(0,5), textcoords='offset points', ha='center', fontsize=10)

    ax.set_xticklabels(models, rotation=45, ha='center')    
    plt.tight_layout()
    plt.savefig('Results/accuracy_comparison_chart.png', dpi=300, bbox_inches='tight')
    print('Generating accuracy comparison chart...')
    plt.show()
    print('Accuracy comparison chart saved to Results/accuracy_comparison_chart.png')

models = ['Rule-Based', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'SVM', 'KNN']
accuracy_results = [1.000, 0.9900, 0.9900, 0.9200, 0.9050, 0.9000]
plot_accuracy_comparison(models, accuracy_results)



# Cross domain mapping visualization - genre availability for each mood
def plot_genre_availability(mapper_df):
    moods = mapper_df['Mood']
    movie_availability = mapper_df['Movie Genres Available']
    colors = ['#4D497A', "#E78FDF", "#E5E88E", '#FF6F61']

    fig, ax = plt.subplots(figsize=(10,6))
    bar = ax.bar(moods, movie_availability, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title('Genre Availability for Each Mood', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Movie Genres Available')

    for x in bar:
        height = x.get_height()
        ax.annotate(f'{height}', xy=(x.get_x() + x.get_width()/2, height), xytext=(0,3), textcoords='offset points', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('Results/genre_availability_chart.png', dpi=300, bbox_inches='tight')
    print('Generating genre availability chart...') 
    plt.show()
    print('Genre availability chart saved to Results/genre_availability_chart.png')



# songs associated with each mood visualization
def plot_songs_per_mood(songs_per_mood):
    moods = list(songs_per_mood.keys())
    num_songs = list(songs_per_mood.values())
    colors = ["#56509D", "#E78FDF", "#24A4C1", '#FF6F61']
    fig,ax = plt.subplots(figsize=(10,6))
    bars = ax.bar(moods, num_songs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title('Number of songs associated with each mood', fontsize=14)
    ax.set_ylabel('Number of Songs')
    
    for i in bars:
        height = i.get_height()
        ax.annotate(f'{height}', xy=(i.get_x() + i.get_width()/2, height), xytext=(0,3), textcoords='offset points', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('Results/songs_per_mood_chart.png', dpi=300, bbox_inches='tight')
    print('Generating songs per mood chart...')
    plt.show()
    print('Songs per mood chart saved to Results/songs_per_mood_chart.png')


songs_per_mood = {
    'Happy': 198,
    'Sad': 152,
    'Energetic': 151,
    'Calm': 499
}
plot_songs_per_mood(songs_per_mood)

# system performance metrics histogram
def system_heatmap_metrics(performance):
    metrcs = ['Average Response Time (ms)', 'Minimum Response Time (ms)', 'Maximum Response Time (ms)', '95th Percentile Response Time (ms)', 'Standard Deviation (ms)', 'Mean Response Time (ms)', 'Success Rate (%)']
    values = [50.48, 50.15, 50.87, 50.76, 0.18, 50.55, 100.0]
    colors = ['#4D497A', "#CB21BD", "#DFE360", "#801309", "#01A698", "#B30101", "#0E40F6"]
    fig, ax = plt.subplots(figsize=(12,6))
    bars = ax.bar(metrcs, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title('System Performance Metrics', fontsize=14, fontweight='bold')      
    ax.set_ylabel('Value')
    ax.set_xticklabels(metrcs, rotation=45, ha='right')
    for i in bars:
        height = i.get_height()
        ax.annotate(f'{height:.2f}', xy=(i.get_x() + i.get_width()/2, height), xytext=(0,3), textcoords='offset points', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('Results/system_performance_metrics_chart.png', dpi=300, bbox_inches='tight')
    print('Generating system performance metrics chart...')
    plt.show()
    print('System performance metrics chart saved to Results/system_performance_metrics_chart.png')
performance = {
    'Average Response Time (ms)': 50.48,
    'Minimum Response Time (ms)': 50.15,
    'Maximum Response Time (ms)': 50.87,    
    '95th Percentile Response Time (ms)': 50.76,
    'Standard Deviation (ms)': 0.18,
    'Mean Response Time (ms)': 50.55,
    'Success Rate (%)': 100.0   
}
system_heatmap_metrics(performance)