import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load and prepare data
df = pd.read_csv('dataset spotify.csv')
features = ['danceability', 'energy', 'valence']
X = df[features]

# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['mood'] = kmeans.fit_predict(X_scaled)

# 4. Name the clusters based on post-analysis interpretation
mood_names = ['Chill', 'Energetic', 'Balanced']
df['mood_label'] = df['mood'].apply(lambda x: mood_names[x])

# 5. Visualize results
plt.figure(figsize=(8, 5))
colors = ['blue', 'red', 'green']

for i in range(3):
    cluster_data = df[df['mood'] == i]
    plt.scatter(
        cluster_data['energy'],
        cluster_data['valence'],
        label=mood_names[i],
        color=colors[i],
        alpha=0.6
    )

plt.xlabel('Energy (Intensity)')
plt.ylabel('Valence (Positiveness)')
plt.title('AI Music Mood Clustering for Adaptive Playlists')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('music_mood_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Show cluster statistics
print("Mood Cluster Analysis:")
print(df['mood_label'].value_counts())

print("\nAverage Features per Mood:")
cluster_stats = df.groupby('mood_label')[features].mean().round(3)
print(cluster_stats)

# 7. Recommendation function
def get_mood_playlist(mood_name, n=5):
    return df[df['mood_label'] == mood_name].head(n)[features + ['liked']]

# 8. Demo recommendations
print("\n🎧 Sample Recommendations:")

print("\nEnergetic Playlist:")
print(get_mood_playlist('Energetic', 3))

print("\nChill Playlist:")
print(get_mood_playlist('Chill', 3))
