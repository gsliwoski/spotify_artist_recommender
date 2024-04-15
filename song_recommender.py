import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import cosine
import json

# Your Spotify App Credentials
creds = json.load(open("spotify/credentials.json"))
SPOTIPY_CLIENT_ID = creds['SPOTIPY_CLIENT_ID']
SPOTIPY_CLIENT_SECRET = creds['SPOTIPY_CLIENT_SECRET']
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'
SCOPE = 'user-read-recently-played'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope=SCOPE))

def find_artists_with_matching_genres(target_genres):
    matched_artists = []
    # Try searching for a perfect match of genres then randomly remove one genre at a time until get at least 100
    while len(target_genres) > 0 and len(matched_artists) < 10:
        print(f"Trying {target_genres}")
        query = " AND ".join([f"genre:\"{genre}\"" for genre in target_genres])
        artist_results = sp.search(q=query, type='artist', limit=10)
        matched_artists.extend([{'id':artist['id'], 'name':artist['name']} for artist in artist_results['artists']['items'] if artist not in matched_artists])
        random_item = random.choice(target_genres)
        target_genres.remove(random_item)
    return matched_artists

def get_top_track_features(artist_id):
    try:
        top_tracks = sp.artist_top_tracks(artist_id, country='US')['tracks']
        if top_tracks:
            top_track_id = top_tracks[0]['id']
            features = sp.audio_features([top_track_id])[0]
            features['track_id'] = top_track_id
            if not features:
                print("Audio features could not be retrieved.")
                features = {}
        else:
            print("No top tracks found for this artist.")
            features = {}
    except Exception as e:
        print(f"Error retrieving top track for artist: {e}")
        features = {}
    return features

def average_cosine_distance(A, B):
    average_distances = []
    for b in B:
        distances = [cosine(b, a) for a in A]
        average_distances.append(np.mean(distances))
    return np.array(average_distances)

results = sp.current_user_recently_played(limit=50)

tracks = []
for idx, item in enumerate(results['items']):
    track = item['track']
    artist_id = track['artists'][0]['id']
    if artist_id in [x['artist_id'] for x in tracks]:
        continue
    elif len(tracks) >= 10:
        break
    features = get_top_track_features(artist_id)
    features['name'] = track['artists'][0]['name']
    features['artist_id'] = artist_id
    tracks.append(features)
    print(f"{idx+1}: {track['artists'][0]['name']} - {track['name']}")

reference_df = pd.DataFrame.from_records(tracks)

artist_features = []
for idx, item in enumerate(results['items']):
    artist_id = item['artist_id']
    artist = sp.artist(artist_id)
    genres = artist['genres']
    if len(genres)<1:
        continue
    matched_artists = find_artists_with_matching_genres([x for x in genres])
    if len(matched_artists) <= 1: #If it only managed to match itself
        continue
    for artist in matched_artists:
        artist_id = artist['id']
        if artist_id in [x['artist_id'] for x in tracks] or artist_id in [x['artist_id'] for x in artist_features]:
            continue
        print(artist['name'])
        features = get_top_track_features(artist_id)
        if len(features.keys())==0:
            continue
        features['artist_id'] = artist_id
        features['name'] = artist['name']
        artist_features.append(features)
artist_df = pd.DataFrame(artist_features)

music_features = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]
sheet_features = ["key", "mode", "tempo"]

reference_df['source'] = 'reference'
artist_df['source'] = 'artist'
df = pd.concat([reference_df, artist_df],ignore_index=True)

all_feats = StandardScaler().fit_transform(df[music_features+sheet_features])
music_feats = StandardScaler().fit_transform(df[music_features])
music_tempo_feats = StandardScaler().fit_transform(df[music_features+["tempo"]])

all_feats_ref = all_feats[df[df.source=="reference"].index]
all_feats_new = all_feats[df[df.source=="artist"].index]

result_distances = average_cosine_distance(all_feats_ref, all_feats_new)
results = pd.DataFrame({"artist": df[df.source=="artist"]['name'].values, "distance": result_distances})
results.sort_values("distance").head(10)

music_feats_ref = music_feats[df[df.source=="reference"].index]
music_feats_new = music_feats[df[df.source=="artist"].index]

result_music_distances = average_cosine_distance(music_feats_ref, music_feats_new)
results_music_distances = pd.DataFrame({"artist": df[df.source=="artist"]['name'].values, "distance": result_music_distances})
results_music_distances.sort_values("distance").head(10)