"""Spotify actions using the Spotipy library.

Requires:
    pip install spotipy

Environment variables (add to your .env or config):
    SPOTIFY_CLIENT_ID
    SPOTIFY_CLIENT_SECRET
    SPOTIFY_REDIRECT_URI  (set this to http://localhost:8888/callback in your Spotify app)

First run will open a browser to authenticate — token is cached after that.
"""
from __future__ import annotations

import os
import traceback
from typing import Any, AsyncIterator

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from .base import Action, ActionContext, get_registry

# Expanded scopes to allow reading private playlists and library (Liked Songs)
SCOPE = (
    "user-read-playback-state "
    "user-modify-playback-state "
    "user-read-currently-playing "
    "streaming "
    "playlist-read-private "
    "user-library-read"
)


def _get_client() -> spotipy.Spotify:
    auth = SpotifyOAuth(
        client_id=os.environ["SPOTIFY_CLIENT_ID"],
        client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8090/callback"),
        scope=SCOPE,
        open_browser=True,
        cache_path=".spotify_token_cache",
    )
    return spotipy.Spotify(auth_manager=auth)


def _get_device_id(sp: spotipy.Spotify, prefer_name: str = "Pi Assistant") -> str | None:
    try:
        devices_payload = sp.devices()
        devices = devices_payload.get("devices", []) if devices_payload else []
    except Exception:
        return None
        
    if not devices:
        return None
    # Prefer the Pi by name, fall back to first available device
    for d in devices:
        if prefer_name.lower() in d.get("name", "").lower():
            return d.get("id")
    return devices[0].get("id")


class SpotifyPlayAction(Action):
    name = "spotify_play"
    description = (
        "Plays a song, artist, or playlist on Spotify. "
        "Can handle specific track names, artists, personal playlists, or 'liked songs'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The general search terms for a song, artist, or playlist.",
            },
            "type": {
                "type": "string",
                "enum": ["track", "artist", "playlist"],
                "description": "The explicit type of content to search for. Default: track.",
            },
            "song": {
                "type": "string",
                "description": "The specific track name if isolated from the user request.",
            },
            "artist": {
                "type": "string",
                "description": "The specific artist name if isolated from the user request.",
            },
            "playlist": {
                "type": "string",
                "description": "The specific playlist name requested by the user (e.g., 'Rock Mix', 'coding chill mix').",
            },
            "playlist_name": {
                "type": "string",
                "description": "Alternative key for the playlist name.",
            },
        },
        "required": [],
    }

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        arguments = arguments or {}
        if not isinstance(arguments, dict):
            yield "Invalid arguments received."
            return

        def safe_str(v: Any) -> str:
            return str(v).strip() if isinstance(v, str) else ""

        song = safe_str(arguments.get("song") or arguments.get("track"))
        artist = safe_str(arguments.get("artist"))
        playlist_name = safe_str(arguments.get("playlist_name") or arguments.get("playlist"))
        raw_query = safe_str(arguments.get("query") or arguments.get("entity") or arguments.get("name"))
        search_type = safe_str(arguments.get("type")).lower()

        if playlist_name and not raw_query:
            raw_query = playlist_name
            search_type = "playlist"

        if not song and not artist and not raw_query:
            rogue_strings = []
            for k, v in arguments.items():
                if k == "type":
                    continue
                k_lower = k.lower()
                if "playlist" in k_lower:
                    search_type = "playlist"
                elif "artist" in k_lower and not artist:
                    search_type = "artist"
                if isinstance(v, str) and v.strip():
                    rogue_strings.append(v.strip())
                elif isinstance(v, (int, float)):
                    rogue_strings.append(str(v))
            if rogue_strings:
                raw_query = " ".join(rogue_strings)

        if search_type not in {"track", "artist", "playlist"}:
            search_type = "playlist" if playlist_name else "track"

        if song or artist:
            if song and artist:
                query = f"track:{song} artist:{artist}"
            elif song:
                query = f"track:{song}"
            else:
                query = f"artist:{artist}"
                search_type = "artist"
        elif raw_query:
            query = raw_query
        else:
            yield "What would you like me to play?"
            return

        try:
            sp = _get_client()
            device_id = _get_device_id(sp)

            if not device_id:
                yield "Spotify is open, but idling. Open your phone or play a track manually once to wake it up."
                return

            q_lower = query.lower()
            if "liked song" in q_lower or "favorite" in q_lower or "favourite" in q_lower:
                try:
                    user_profile = sp.current_user()
                    user_id = user_profile.get("id") if user_profile else None
                    if not user_id:
                        yield "Could not retrieve your Spotify user ID."
                        return
                    collection_uri = f"spotify:user:{user_id}:collection"
                    sp.start_playback(device_id=device_id, context_uri=collection_uri)
                    yield "Playing your Liked Songs library."
                    return
                except Exception as liked_err:
                    yield f"Failed to stream Liked Songs: {liked_err}"
                    return

            if search_type == "playlist":
                clean_target = " ".join(q_lower.replace("my playlist", "").replace("the playlist", "").replace("playlist", "").split()).strip()
                all_my_playlists = []
                limit = 50
                offset = 0
                while True:
                    try:
                        playlist_page = sp.current_user_playlists(limit=limit, offset=offset)
                    except Exception:
                        playlist_page = None
                    if not playlist_page or not isinstance(playlist_page, dict):
                        break
                    playlists = playlist_page.get("items")
                    if not isinstance(playlists, list) or not playlists:
                        break
                    for p in playlists:
                        if p and isinstance(p, dict) and p.get("name") is not None:
                            all_my_playlists.append(p)
                    if len(playlists) < limit:
                        break
                    offset += limit

                for p in all_my_playlists:
                    name_str = p.get("name", "")
                    if " ".join(name_str.lower().split()).strip() == clean_target:
                        sp.start_playback(device_id=device_id, context_uri=p.get("uri"))
                        yield f"Playing your playlist {name_str}."
                        return

                for p in all_my_playlists:
                    name_str = p.get("name", "")
                    clean_name = " ".join(name_str.lower().split()).strip()
                    if clean_target in clean_name or clean_name in clean_target:
                        sp.start_playback(device_id=device_id, context_uri=p.get("uri"))
                        yield f"Playing your playlist {name_str}."
                        return

            if not (song or artist):
                cleaned = query.lower()
                for filler in [" by a ", " by ", "my playlist ", "the playlist ", "playlist "]:
                    cleaned = cleaned.replace(filler, " ")
                query = cleaned.strip()

            try:
                results = sp.search(q=query, type=search_type, limit=10)
            except Exception as search_err:
                yield f"Spotify search failed: {search_err}"
                return

            if not results or not isinstance(results, dict) or not results.get(search_type + "s"):
                yield f"I couldn't find any {search_type} matching '{query}'."
                return

            items = results[search_type + "s"].get("items", [])
            if not items:
                yield f"I couldn't find any {search_type} matching '{query}'."
                return

            uri, chosen_item = None, None
            for candidate in items:
                if not candidate: continue
                c_uri = candidate.get("uri")
                if c_uri and not ("collab" in c_uri or ("mix" in c_uri and not candidate.get("images"))):
                    uri, chosen_item = c_uri, candidate
                    break

            if not uri and items:
                chosen_item = items[0]
                uri = chosen_item.get("uri")

            if not uri:
                yield "Found results, but content was unplayable."
                return

            if search_type == "track":
                sp.start_playback(device_id=device_id, uris=[uri])
                yield f"Playing track: {chosen_item.get('name', 'Unknown Track')}."
            elif search_type == "artist":
                sp.start_playback(device_id=device_id, context_uri=f"spotify:artist:{chosen_item.get('id')}:top-tracks")
                yield f"Playing top tracks by {chosen_item.get('name', 'Unknown Artist')}."
            elif search_type == "playlist":
                sp.start_playback(device_id=device_id, context_uri=uri)
                yield f"Playing playlist: {chosen_item.get('name', 'Unknown Playlist')}."

        except Exception as e:
            yield f"Spotify error: {e}"


class SpotifyShuffleAction(Action):
    name = "spotify_shuffle"
    description = "Toggles shuffle on or off for Spotify playback."
    parameters = {
        "type": "object",
        "properties": {
            "state": {"type": "boolean", "description": "True to turn shuffle on, False to turn it off."}
        },
    }

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        try:
            sp = _get_client()
            device_id = _get_device_id(sp)
            if not device_id:
                yield "No active Spotify device found."
                return
            state = arguments.get("state", True)
            sp.shuffle(state=state, device_id=device_id)
            yield f"Shuffle turned {'on' if state else 'off'}."
        except Exception as e:
            yield f"Spotify error: {e}"


class SpotifyPauseAction(Action):
    name = "spotify_pause"
    description = "Pauses Spotify playback."
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        try:
            sp = _get_client()
            device_id = _get_device_id(sp)
            # We explicitly pass device_id to target the Pi even if it's idling
            sp.pause_playback(device_id=device_id)
            yield "Paused."
        except Exception as e:
            if "NO_ACTIVE_DEVICE" in str(e):
                yield "Nothing is currently playing."
            else:
                yield f"Spotify error: {e}"


class SpotifyResumeAction(Action):
    name = "spotify_resume"
    description = "Resumes Spotify playback."
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        try:
            sp = _get_client()
            device_id = _get_device_id(sp)
            sp.start_playback(device_id=device_id)
            yield "Resuming."
        except Exception as e:
            if "NO_ACTIVE_DEVICE" in str(e):
                yield "Spotify is idle. Ask me to play a specific song to wake it up."
            else:
                yield f"Spotify error: {e}"


class SpotifySkipAction(Action):
    name = "spotify_skip"
    description = "Skips to the next track on Spotify."
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        try:
            sp = _get_client()
            device_id = _get_device_id(sp)
            sp.next_track(device_id=device_id)
            yield "Skipped."
        except Exception as e:
            yield f"Spotify error: {e}"


class SpotifyVolumeAction(Action):
    name = "spotify_volume"
    description = "Sets or adjusts the Spotify playback volume."
    parameters = {
        "type": "object",
        "properties": {
            "level": {"type": "integer", "description": "Absolute volume level 0 to 100."},
            "direction": {"type": "string", "enum": ["up", "down"]},
        },
    }

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        try:
            sp = _get_client()
            device_id = _get_device_id(sp)
            direction = arguments.get("direction")
            level = arguments.get("level")

            if direction:
                current = sp.current_playback()
                vol = current["device"].get("volume_percent", 50) if current and current.get("device") else 50
                level = min(100, vol + 20) if direction == "up" else max(0, vol - 20)

            if level is None:
                yield "What volume level would you like?"
                return

            final_volume = max(0, min(100, int(level)))
            sp.volume(final_volume, device_id=device_id)
            yield f"Volume set to {final_volume}%."
        except Exception as e:
            yield f"Spotify error: {e}"


class SpotifyNowPlayingAction(Action):
    name = "spotify_now_playing"
    description = "Tells the user what song is currently playing on Spotify."
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        try:
            current = _get_client().current_playback()
            if not current or not current.get("item"):
                yield "Nothing is playing right now."
                return
            track = current["item"]
            yield f"Playing {track['name']} by {track['artists'][0]['name']}."
        except Exception as e:
            yield f"Spotify error: {e}"


def register_spotify_actions() -> None:
    reg = get_registry()
    for cls in (
        SpotifyPlayAction,
        SpotifyPauseAction,
        SpotifyResumeAction,
        SpotifySkipAction,
        SpotifyVolumeAction,
        SpotifyNowPlayingAction,
        SpotifyShuffleAction,
    ):
        if reg.get(cls.name) is None:
            reg.register(cls())