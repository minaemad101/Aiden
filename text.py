import spotipy
import os
import dotenv
import orjson
import random
from spotipy.oauth2 import SpotifyOAuth
from rich import print
from methods import *
import asyncio
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from src.nlp import *



dotenv.load_dotenv()

with open('settings.json') as f:
    settings = orjson.loads(f.read())
presets = settings["presets"]
f.close()

scope = f"ugc-image-upload, user-read-playback-state, user-modify-playback-state, user-follow-modify, user-read-private, user-follow-read, user-library-modify, user-library-read, streaming, user-read-playback-position, app-remote-control, user-read-email, user-read-currently-playing, user-read-recently-played, playlist-modify-private, playlist-read-collaborative, playlist-read-private, user-top-read, playlist-modify-public"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, client_id=os.getenv("SPOTIFY_CLIENT_ID"), client_secret=os.getenv(
    "SPOTIFY_CLIENT_SECRET"), redirect_uri="http://localhost:8888/callback"), requests_timeout=300)


def default_processing(words):
    action = words[0]
    name = " ".join(words[1:])
    try:
        if action == "current":
            if name == "song":
                """ Display the current song playing. """
                track = asyncio.run(get_current_song(spotify=sp))
                print(
                    f"[bold deep_sky_blue2]Current track:[/bold deep_sky_blue2] [italic spring_green3]{track}[/italic spring_green3]")
    except Exception as action_exception:
        print(
            f"[italic red]Could not underst{action_exception}and.[/italic red]")
    try:
        if action == 'go':
            if name == 'back':
                """ Go Back to previous song. """
                asyncio.run(play_previous_song(spotify=sp))
    except Exception as e:
        print(f"[italic red]{e}[/italic red]")
    try:
        if action == "play":
            if name == "random":
                tracks = asyncio.run(get_user_saved_tracks(spotify=sp))
                random_track = random.choice(tracks)
                uri = asyncio.run(get_track_uri(spotify=sp, name=random_track))
                asyncio.run(play_track(spotify=sp, uri=uri))
                print(
                    f"[bold deep_sky_blue2]Playing track:[/bold deep_sky_blue2] [italic spring_green3]{random_track}[/italic spring_green3]")
            else:
                uri = asyncio.run(get_track_uri(spotify=sp, name=name))
                asyncio.run(play_track(spotify=sp, uri=uri))
                print(
                    f"[bold deep_sky_blue2]Playing track:[/bold deep_sky_blue2] [italic spring_green3]{name}[/italic spring_green3]")

        if action == "album":
            uri = asyncio.run(get_album_uri(spotify=sp, name=name))
            asyncio.run(play_album(spotify=sp, uri=uri))
            print(
                f"[bold deep_sky_blue2]Playing album:[/bold deep_sky_blue2] [italic spring_green3]{name}[/italic spring_green3]")

        if action == "artist":
            if name == "random":
                random_artist = random.choice(
                    get_user_followed_artists(spotify=sp))
                uri = asyncio.run(get_artist_uri(
                    spotify=sp, name=random_artist))
                asyncio.run(play_artist(spotify=sp, uri=uri))
                print(
                    f"[bold deep_sky_blue2]Playing artist:[/bold deep_sky_blue2] [italic spring_green3]{random_artist}[/italic spring_green3]")
            else:
                uri = asyncio.run(get_artist_uri(spotify=sp, name=name))
                asyncio.run(play_artist(spotify=sp, uri=uri))
                print(
                    f"[bold deep_sky_blue2]Playing artist:[/bold deep_sky_blue2] [italic spring_green3]{name}[/italic spring_green3]")

        if action == "playlist":
            playlists, playlist_ids = asyncio.run(
                get_user_playlists(spotify=sp))
            if name.lower() in playlists:
                for i in range(len(playlists)):
                    if name.lower() == playlists[i].lower():
                        id = playlist_ids[i]
                        asyncio.run(play_playlist(spotify=sp, playlist_id=id))
                        print(
                            f"[bold deep_sky_blue2]Playing playlist:[/bold deep_sky_blue2] [italic spring_green3]{name}[/italic spring_green3]")
            else:
                print("[italic red]Could not find playlist.[/italic red]")
                return
        elif action == "volume":
            t = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}  # dictionary for volume
            if name in t:
                """ For some reason speech recognition return 1 - 10 as strings, so we need to convert them to ints."""
                volume = t[name]
                asyncio.run(change_volume(spotify=sp, volume=volume))
                print(
                    f"[bold deep_sky_blue2]Volume set to:[/bold deep_sky_blue2] [italic spring_green3]{volume}[/italic spring_green3]")
            else:
                """ If volume is not in dictionary, tthen return it as is."""
                volume = int(name)
                asyncio.run(change_volume(spotify=sp, volume=volume))
                print(
                    f"[bold deep_sky_blue2]Volume set to:[/bold deep_sky_blue2] [italic spring_green3]{volume}%[/italic spring_green3]")

        elif action == "shuffle":
            state = name
            asyncio.run(shuffle(spotify=sp, state=state))

    except InvalidSearchError:
        print(f"[italic red]Could not find {name}. Try again.[/italic red]")


def text(command):
    command = command.lower()
    print(f"[medium_purple3]{command}[/medium_purple3]")
    words = command.split()
    if len(words) < 1:
        print(f"[italic red]Could not understand.[/italic red]")
        return
    elif len(words) == 1:
        if words[0] == "next":
            asyncio.run(next_track(spotify=sp))
        elif words[0] == "pause":
            asyncio.run(pause_track(spotify=sp))
        elif words[0] == "resume":
            asyncio.run(resume_track(spotify=sp))
        elif words[0] == 'back':
            asyncio.run(play_previous_song(spotify=sp))
        elif words[0] == "quit":
            asyncio.run(exit_application())
            return
        elif words[0] == "repeat":
            asyncio.run(repeat_track(spotify=sp))
        else:
            print(f"[italic red]Command not recognized.[/italic red]")
            return
    elif "next" in words:
        asyncio.run(next_track(spotify=sp))
    elif "pause" in words:
        asyncio.run(pause_track(spotify=sp))
    elif "resume" in words:
        asyncio.run(resume_track(spotify=sp))
    elif 'back' in words or "previous" in words:
        asyncio.run(play_previous_song(spotify=sp))
    elif "repeat" in words:
        asyncio.run(repeat_track(spotify=sp))
    else:
        intent = classify(command)
        print(intent)
        if intent == 45:
            entities = ner(command)
            song = {i for i in entities if entities[i] == "SONG"}
            album = {i for i in entities if entities[i] == "ALBUM"}
            artist = {i for i in entities if entities[i] == "ARTIST"}
            playlist = {i for i in entities if entities[i] == "PLAYLIST"}
            genre = {i for i in entities if entities[i] == "GENRE"}
            if (len(song)):
                song = list(song)
                uri = asyncio.run(get_track_uri(spotify=sp, name=song[0]))
                print(uri)
                asyncio.run(play_track(spotify=sp, uri=uri))
                print(
                    f"[bold deep_sky_blue2]Playing track:[/bold deep_sky_blue2] [italic spring_green3]{song[0]}[/italic spring_green3]")
            elif (len(album)):
                album = list(album)
                uri = asyncio.run(get_album_uri(spotify=sp, name=album[0]))
                asyncio.run(play_album(spotify=sp, uri=uri))
                print(
                    f"[bold deep_sky_blue2]Playing album:[/bold deep_sky_blue2] [italic spring_green3]{album[0]}[/italic spring_green3]")
            elif (len(artist)):
                artist = list(artist)
                uri = asyncio.run(get_artist_uri(spotify=sp, name=artist[0]))
                asyncio.run(play_artist(spotify=sp, uri=uri))
                print(
                    f"[bold deep_sky_blue2]Playing artist:[/bold deep_sky_blue2] [italic spring_green3]{artist[0]}[/italic spring_green3]")
            elif (len(playlist)):
                playlist = list(playlist)
                name = playlist[0]
                playlists, playlist_ids = asyncio.run(
                    get_user_playlists(spotify=sp))
                if name.lower() in playlists:
                    for i in range(len(playlists)):
                        if name.lower() == playlists[i].lower():
                            id = playlist_ids[i]
                            asyncio.run(play_playlist(
                                spotify=sp, playlist_id=id))
                            print(
                                f"[bold deep_sky_blue2]Playing playlist:[/bold deep_sky_blue2] [italic spring_green3]{name}[/italic spring_green3]")

                else:
                    print("[italic red]Could not find playlist.[/italic red]")
            elif (len(genre)):
                genre = list(genre)
                uri, type = asyncio.run(
                    get_track_uri_for_genre(spotify=sp, genre=genre[0]))
                if type == "track":
                    asyncio.run(play_track(spotify=sp, uri=uri))
                elif type == "album":
                    asyncio.run(play_album(spotify=sp, uri=uri))
                elif type == "artist":
                    asyncio.run(play_artist(spotify=sp, uri=uri))
                print(
                    f"[bold deep_sky_blue2]Playing genre:[/bold deep_sky_blue2] [italic spring_green3]{genre[0]}[/italic spring_green3]")

        elif intent == 14:
            current_volume = sp.current_playback()['device']['volume_percent']
            new_volume = min(current_volume + 10, 100)
            asyncio.run(change_volume(spotify=sp, volume=new_volume))
            print(
                f"[bold deep_sky_blue2]Volume set to:[/bold deep_sky_blue2] [italic spring_green3]{80}%[/italic spring_green3]")
        elif intent == 35:
            current_volume = sp.current_playback()['device']['volume_percent']
            new_volume = max(current_volume - 10, 0)
            asyncio.run(change_volume(spotify=sp, volume=new_volume))
            print(
                f"[bold deep_sky_blue2]Volume set to:[/bold deep_sky_blue2] [italic spring_green3]{20}%[/italic spring_green3]")
        elif intent == 46:
            asyncio.run(change_volume(spotify=sp, volume=0))
            print(
                f"[bold deep_sky_blue2]Volume set to:[/bold deep_sky_blue2] [italic spring_green3]{20}%[/italic spring_green3]")
        else:
            default_processing(words)
        return


print("[bold deep_sky_blue2]Goodbye![/bold deep_sky_blue2]")
