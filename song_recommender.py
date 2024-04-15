import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import cosine
import json
from tqdm import tqdm
import argparse

sp = None

MUSIC_FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]
SHEET_FEATURES = ["key", "mode", "tempo"]

def initialize_spotify_client(credentials_file):
    global sp
    creds = json.load(open(credentials_file))
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
        matched_artists.extend([{'id':artist['id'], 'name':artist['name'], 'artist_url':artist['external_urls']['spotify']} for artist in artist_results['artists']['items'] if artist not in matched_artists])
        random_item = random.choice(target_genres)
        target_genres.remove(random_item)
    return matched_artists

def get_top_track_features(artist_id):
    try:
        top_tracks = sp.artist_top_tracks(artist_id, country='US')['tracks']
        if top_tracks:
            top_track_id = top_tracks[0]['id']
            top_track_name = top_tracks[0]['name']
            features = sp.audio_features([top_track_id])[0]
            features['track_id'] = top_track_id
            features['track_name'] = top_track_name
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

def get_recently_played(limit=50, selected_artists = ""):
    if len(selected_artists) == 0:
        print("Getting most recently played artists")
        results = sp.current_user_recently_played(limit=limit)
    else:
        selected_artists = [x.strip() for x in selected_artists.split(",")]
        print(f"Using supplied list of artists (first 10 artists only)")
        results = {'items':[]}
        for artist in selected_artists[:10]:
            artist_objects = sp.search(q=f"artist: {artist}", type='artist')
            try:
                artist_objects = sorted([x for x in artist_objects['artists']['items'] if x['name'].lower() == artist.lower()],
                                        key= lambda x: x['popularity'], reverse=True)
            except KeyError:
                artist_objects = []
            if len(artist_objects) == 0:
                print(f"{artist} not found")
                continue
            elif len(artist_objects) > 1:
                print(f"Multiple artist_id found for {artist}, selecting the most popular artist_id in the list.")
            artist_id = artist_objects[0]['id']
            try:
                artist_url = artist_objects[0]['external_urls']['spotify']
            except KeyError:
                artist_url = ""
            results['items'].append({
                'track': {
                    'artists': [{'id': artist_id,
                                  'name': artist,
                                  'external_urls': {'spotify': artist_url}}]}})
    tracks = []
    for idx, item in enumerate(results['items']):
        try:
            track = item['track']
            artist_id = track['artists'][0]['id']
        except KeyError as e:
            print(f"Failed index {idx}:")
            print(e)
            continue
        if artist_id in [x['artist_id'] for x in tracks]:
            continue
        elif len(tracks) >= 10:
            break
        features = get_top_track_features(artist_id)
        if len(features.keys()) == 0:
            print(f"No features found for artist: {artist_id}")
            continue
        try:
            features['name'] = track['artists'][0]['name']
        except KeyError as e:
            print(f"Failed to get artists name for {artist_id}:")
            print(e)
            features['name'] = np.nan
        features['artist_id'] = artist_id
        try:
            features['artist_url'] = track['artists'][0]['external_urls']['spotify']
        except KeyError as e:
            print(f"Failed to get URL for artist {artist_id}:")
            print(e)
            features['artist_url'] = np.nan
        tracks.append(features)
        print(f"{idx+1}: {features['name']} - {features['track_name']}")
    return tracks

def get_matching_artists(tracks):
    artist_features = []
    for idx, item in tqdm(enumerate(tracks)):
        artist_id = item['artist_id']
        artist = sp.artist(artist_id)
        genres = artist.get('genres',[])
        if len(genres)<1:
            continue
        matched_artists = find_artists_with_matching_genres([x for x in genres])
        if len(matched_artists) <= 1: #If it only managed to match itself
            continue
        for artist in matched_artists:
            artist_id = artist.get('id',"")
            if artist_id == "" or artist_id in [x['artist_id'] for x in tracks+artist_features]:
                continue
            #print(artist.get('name', 'UNKNOWN ARTIST'))
            features = get_top_track_features(artist_id)
            if len(features.keys())==0:
                continue
            features['artist_id'] = artist_id
            features['name'] = artist['name']
            features['artist_url'] = artist['artist_url']
            artist_features.append(features)
    return artist_features

def get_closest_artists(df, features):
    all_feats = StandardScaler().fit_transform(df[features])
    all_feats_ref = all_feats[df[df.source=="reference"].index]
    all_feats_new = all_feats[df[df.source=="artist"].index]
    result_distances = average_cosine_distance(all_feats_ref, all_feats_new)
    results = pd.DataFrame({"artist": df[df.source == "artist"]['name'].values, "distance": result_distances,
                            'artist_url': df[df.source == "artist"]['artist_url']})
    return results.sort_values("distance").head(10)

#music_feats = StandardScaler().fit_transform(df[music_features])
#music_tempo_feats = StandardScaler().fit_transform(df[music_features+["tempo"]])
# music_feats_ref = music_feats[df[df.source=="reference"].index]
# music_feats_new = music_feats[df[df.source=="artist"].index]
# result_music_distances = average_cosine_distance(music_feats_ref, music_feats_new)
# results_music_distances = pd.DataFrame({"artist": df[df.source=="artist"]['name'].values, "distance": result_music_distances})
# results_music_distances.sort_values("distance").head(10)

def main():
    parser = argparse.ArgumentParser(description="Spotify artist recommender. Requires a JSON with spotify credentials "
                                                 "(see credentials.json.example). Can also take a comma separated list "
                                                 "of artists instead of looking up last played.")
    parser.add_argument('--creds', type=str, help='Path to credentials json file', required=True)
    parser.add_argument('--artists', type=str, help='Comma separated list of artists', default="")
    args = parser.parse_args()

    print("Initializing Spotify")
    initialize_spotify_client(args.creds)
    tracks = get_recently_played(selected_artists=args.artists)
    print("Getting reference features")
    reference_df = pd.DataFrame.from_records(tracks)
    print("Getting matching artist features")
    artist_features = get_matching_artists(tracks)
    artist_df = pd.DataFrame(artist_features)
    reference_df['source'] = 'reference'
    artist_df['source'] = 'artist'
    df = pd.concat([reference_df, artist_df],ignore_index=True)
    closest_artists = get_closest_artists(df, MUSIC_FEATURES+SHEET_FEATURES)
    print(closest_artists)
    closest_artists.to_csv("closest_artists.csv", index=False)

if __name__ == "__main__":
    main()