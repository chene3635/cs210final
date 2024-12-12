import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# We make a sql database yuhhhhhhh
conn = sqlite3.connect('songs_database.db')
cursor = conn.cursor()

# Making a table for SONGS
cursor.execute('''
CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY,
    track TEXT,
    artist TEXT,
    danceability REAL,
    energy REAL,
    tempo REAL,
    loudness REAL,
    speechiness REAL,
    acousticness REAL,
    instrumentalness REAL,
    liveness REAL,
    valence REAL,
    artist_encoded REAL,
    track_encoded REAL,
    chorus_hit REAL,
    sections REAL,
    duration_ms REAL,
    mode REAL,
    key REAL,
    time_signature REAL
)
''')

# Loading Dataset for 2010 songs and 1990 songs
df_10s = pd.read_csv('dataset-of-10s.csv')
df_90s = pd.read_csv('dataset-of-90s.csv')

# Combine these datasets so we can compare both of the songs
df = pd.concat([df_10s, df_90s], ignore_index=True)
df.drop(columns=['uri', 'target'], inplace=True)

# Data Preprossessing to encode the artist and track columns
encoder = LabelEncoder()
df['artist_encoded'] = encoder.fit_transform(df['artist'])
df['track_encoded'] = encoder.fit_transform(df['track'])

# Normalize
features = ['danceability', 'energy', 'tempo', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'artist_encoded', 'track_encoded',
            'chorus_hit', 'sections', 'duration_ms', 'mode', 'key', 'time_signature']

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Put the data into the database
df.to_sql('songs', conn, if_exists='replace', index=False)

# Scale the data
sScaler = StandardScaler()
scaled_features = sScaler.fit_transform(df[features])

# Embedding the data with PCAS
pcas = PCA(n_components=10)
embeddings = pcas.fit_transform(scaled_features)

# Calculating cosine simularity
cosineSimilar = cosine_similarity(embeddings)

# MEAT of the project (recommends the songs)
def recommendSongs_MachineLearning(song_name, df, similarity_matrix, top_n=5):
    
    # Normalize the song name and dataset for comparison
    song_name = song_name.strip().lower()
    df['track'] = df['track'].str.strip().str.lower()

    # finding song within database
    try:
        song_idx = df[df['track'] == song_name].index[0]
    except IndexError:
        return f"Song '{song_name}' not found in the dataset."

    # Display details of the input song
    print(f"Input Song:\nTrack: {df.iloc[song_idx]['track']}\nArtist: {df.iloc[song_idx]['artist']}\n")

    # Calculate similarity scores
    similarity_scores = list(enumerate(similarity_matrix[song_idx]))

    # Sort similarity scores in descending order
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Generate recommendations
    seen_tracks = set()
    recommended_indices = []

    for idx, _ in similarity_scores:
        track_name = df.iloc[idx]['track']
        artist_name = df.iloc[idx]['artist']

        # Avoid duplicate recommendations or recommending the input song
        if track_name not in seen_tracks and (track_name, artist_name) != (song_name, df.iloc[song_idx]['artist']):
            recommended_indices.append(idx)
            seen_tracks.add(track_name)

        
        if len(recommended_indices) >= top_n:
            break

    # Create a DataFrame of recommendations
    recommendations = df.iloc[recommended_indices][['track', 'artist']].drop_duplicates()
    return recommendations.reset_index(drop=True)


# Function to plot the similarity between an input song and a recommended song
def plot_feature_similarity(song_name, recommended_song, df, features):
    try:
        input_song = df.loc[df['track'].str.strip().str.lower() == song_name.strip().lower()].iloc[0]
        recommended_song_data = df.loc[df['track'].str.strip().str.lower() == recommended_song.strip().lower()].iloc[0]
    except IndexError:
        print("Error: One or both songs not found in the dataset.")
        return

    # Prepare data for the radar plot
    labels = features
    input_values = input_song[features].values
    recommended_values = recommended_song_data[features].values

    # Ensure the radar plot is a closed shape
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    input_values = np.concatenate((input_values, [input_values[0]]))
    recommended_values = np.concatenate((recommended_values, [recommended_values[0]]))

    # Create the radar plot
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, input_values, label=f"Input Song: {song_name}", linewidth=2)
    ax.plot(angles, recommended_values, label=f"Recommended: {recommended_song}", linewidth=2)

    ax.fill(angles, input_values, alpha=0.25)
    ax.fill(angles, recommended_values, alpha=0.25)

    # Configure plot aesthetics
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.title("Feature Similarity", fontsize=14)
    plt.show()

# Example usage of the plot_feature_similarity function
if __name__ == "__main__":
    user_input = input("Enter the name of a song: ").strip()

    # Retrieve recommendations using your custom recommendation function
    recommendations = recommendSongs_MachineLearning(user_input, df, cosineSimilar)

    if isinstance(recommendations, pd.DataFrame):
        recommendations.reset_index(drop=True, inplace=True)
        print("Recommended Songs:")
        print(recommendations)

        if not recommendations.empty:
            recommended_track = recommendations.iloc[0]['track']
            plot_feature_similarity(user_input, recommended_track, df, features)
    else:
        print("No recommendations found or an error occurred:", recommendations)
# Close the database
conn.close()
