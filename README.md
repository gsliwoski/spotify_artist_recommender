A hobby project.

WORK IN PROGRESS

Recommends artists based on the features of the top songs of the last 10 artists you listened to (could be less it currently only goes as far back as 50 tracks looking for unique artists.

Either it uses the most recently played tracks or you can supply a list of artists to use instead.

Usage:

To use the most recent songs you've listened to:

python --creds credentials.json.example

To use a list of artists instead:

python --creds credentials.json.example --artists "The Tallest Man On Earth, Ben Woodward, JJ Heller, Cornelis Vreeswijk"

To get credentials:

1. Go to https://developer.spotify.com/
2. Log in
3. Click on profile and select Dashboard
4. If haven't already, click Create app
5. Click on your app
6. Click on Settings
7. Get Client ID and Client Secret