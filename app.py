import gradio as gr
import pandas as pd
import sys
sys.path.extend(["."])
import artist_recommender

def export_csv(ad_output):
    ad_output.to_csv("output.csv", index=False)
    return gr.File(value="output.csv", visible=True)

def get_artist_recommendations(client_id, client_secret, client_redirect_uri, artist_list, create_playlist):
    print("Initializing Spotify")
    creds = {
        "SPOTIPY_CLIENT_ID": client_id,
        "SPOTIPY_CLIENT_SECRET": client_secret,
        "SPOTIPY_REDIRECT_URI": client_redirect_uri
    }
    artist_recommender.initialize_spotify_client(creds, isfile=False)
    sp = artist_recommender.sp
    try:
        _ = sp.current_user()
    except:
        print("Failed to initialize Spotify, are credentials correct?")
        sys.exit()
    tracks = artist_recommender.get_recently_played(selected_artists=artist_list)
    print("Getting reference features")
    reference_df = pd.DataFrame.from_records(tracks)
    print("Getting matching artist features")
    artist_features = artist_recommender.get_matching_artists(tracks)
    artist_df = pd.DataFrame(artist_features)
    reference_df['source'] = 'reference'
    artist_df['source'] = 'artist'
    df = pd.concat([reference_df, artist_df],ignore_index=True)
    closest_artists = artist_recommender.get_closest_artists(df, artist_recommender.MUSIC_FEATURES+artist_recommender.SHEET_FEATURES)
    closest_artists.to_csv("closest_artists.csv", index=False)
    if create_playlist:
        artist_recommender.generate_playlist(closest_artists, artist_list)
    return closest_artists

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Spotify Artist Recommender
    """
    )
    client_id = gr.Textbox(label="SPOTIFY_CLIENT_ID", value="")
    client_secret = gr.Textbox(label="SPOTIFY_CLIENT_SECRET", value="")
    client_redirect_uri = gr.Textbox(label="SPOTIFY_REDIRECT_URI", value="")
    artist_list = gr.Textbox(label="Arist list (Optional). Leave blank to use your recent activity. Otherwisr a CSV of artists", value="")
    create_playlist = gr.Checkbox(label="Generate a playlist", value=True)
    button = gr.Button("Get Artist Recommendations")
    output1 = gr.DataFrame(headers=['artist', 'artist_id', 'distance', 'artist_url'], interactive=False, wrap=True)
    button.click(fn=get_artist_recommendations,
             inputs=[client_id, client_secret, client_redirect_uri, artist_list, create_playlist],
             outputs=output1)

    export_button = gr.Button("Export Recommendations")
    csv = gr.File(interactive=False, visible=False)
    export_button.click(fn=export_csv, inputs=[output1], outputs=[csv])

    if __name__ == '__main__':
        demo.launch()