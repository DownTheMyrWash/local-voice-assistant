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
    devices = sp.devices().get("devices", [])
    if not devices:
        return None
    # Prefer the Pi by name, fall back to first active device
    for d in devices:
        if prefer_name.lower() in d["name"].lower():
            return d["id"]
    return None


import traceback
from typing import Any, AsyncIterator


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
        # -----------------------------
        # 0. Normalize arguments safely
        # -----------------------------
        arguments = arguments or {}
        if not isinstance(arguments, dict):
            yield "Invalid arguments received."
            return

        def safe_str(v: Any) -> str:
            return str(v).strip() if isinstance(v, str) else ""

        # -----------------------------
        # 1. Capture split parameters
        # -----------------------------
        song = safe_str(arguments.get("song") or arguments.get("track"))
        artist = safe_str(arguments.get("artist"))
        playlist_name = safe_str(arguments.get("playlist_name") or arguments.get("playlist"))

        # -----------------------------
        # 2. Capture standard single-string variations
        # -----------------------------
        raw_query = safe_str(
            arguments.get("query")
            or arguments.get("entity")
            or arguments.get("name")
        )

        search_type = safe_str(arguments.get("type")).lower()

        # If a playlist parameter was provided, force the search type to playlist
        if playlist_name and not raw_query:
            raw_query = playlist_name
            search_type = "playlist"

        # -----------------------------
        # 3. SMART CATCH-ALL (Detects hidden playlist/artist indicators in keys)
        # -----------------------------
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

        # Sanitize / Fallback for search_type
        if search_type not in {"track", "artist", "playlist"}:
            if playlist_name:
                search_type = "playlist"
            else:
                search_type = "track"

        # -----------------------------
        # 4. Build final query
        # -----------------------------
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

        # -----------------------------
        # 5. Spotify API operations
        # -----------------------------
        try:
            sp = _get_client()

            # -----------------------------
            # Get devices safely (Prevent AttributeError if sp.devices() is None)
            # -----------------------------
            try:
                devices_payload = sp.devices()
            except Exception:
                devices_payload = None

            devices = []
            if devices_payload and isinstance(devices_payload, dict):
                devices = devices_payload.get("devices") or []
            
            if not isinstance(devices, list):
                devices = []

            if not devices:
                yield "Spotify is open, but idling. Play or unpause a song manually once to wake up the connection."
                return

            device_id = _get_device_id(sp)
            if not device_id and devices:
                device_id = devices[0].get("id")

            if not device_id:
                yield "Unable to determine a valid Spotify playback device."
                return

            # -----------------------------
            # SPECIAL CASE: Liked Songs (Native Collection Streaming)
            # -----------------------------
            q_lower = query.lower()
            if "liked song" in q_lower or "favorite" in q_lower or "favourite" in q_lower:
                try:
                    # Fetch your current user profile to get your exact Spotify Username
                    user_profile = sp.current_user()
                    user_id = user_profile.get("id") if user_profile else None
                    
                    if not user_id:
                        yield "Could not retrieve your Spotify user ID to load your collection."
                        return
                        
                    # Target the native Liked Songs collection context URI
                    collection_uri = f"spotify:user:{user_id}:collection"
                    
                    # Stream the entire collection directly
                    sp.start_playback(device_id=device_id, context_uri=collection_uri)
                    yield "Playing your Liked Songs library."
                    return
                    
                except Exception as liked_err:
                    yield f"Failed to stream Liked Songs collection natively: {liked_err}"
                    return

            # -----------------------------
            # SPECIAL CASE: User Playlists (Hyper-Aggressive Search & Match)
            # -----------------------------
            if search_type == "playlist":
                # Clean up the user's voice query thoroughly
                clean_target = " ".join(q_lower.replace("my playlist", "").replace("the playlist", "").replace("playlist", "").split()).strip()
                
                all_my_playlists = []
                limit = 50
                offset = 0
                
                # Step 1: Fetch user's library playlists
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

                # PASS 1: Clean Exact Match
                for p in all_my_playlists:
                    name_str = p.get("name", "")
                    clean_name = " ".join(name_str.lower().split()).strip()
                    if clean_name == clean_target:
                        p_uri = p.get("uri")
                        if p_uri:
                            sp.start_playback(device_id=device_id, context_uri=p_uri)
                            yield f"Playing your playlist {name_str}."
                            return

                # PASS 2: Cross-Substring Match (Handles variations or partial names)
                for p in all_my_playlists:
                    name_str = p.get("name", "")
                    clean_name = " ".join(name_str.lower().split()).strip()
                    if clean_target in clean_name or clean_name in clean_target:
                        p_uri = p.get("uri")
                        if p_uri:
                            sp.start_playback(device_id=device_id, context_uri=p_uri)
                            yield f"Playing your playlist {name_str}."
                            return

                # PASS 3: Direct API Search Fallback (Forces Spotify to look up unindexed items)
                try:
                    search_results = sp.search(q=clean_target, type="playlist", limit=20)
                    if search_results and isinstance(search_results, dict):
                        items = search_results.get("playlists", {}).get("items", [])
                        for p in items:
                            if p and isinstance(p, dict):
                                name_str = p.get("name", "")
                                clean_name = " ".join(name_str.lower().split()).strip()
                                
                                # If it matches or heavily overlaps, grab it
                                if clean_target in clean_name or clean_name in clean_target:
                                    p_uri = p.get("uri")
                                    if p_uri:
                                        sp.start_playback(device_id=device_id, context_uri=p_uri)
                                        yield f"Playing playlist: {name_str}."
                                        return
                except Exception:
                    pass
            # -----------------------------
            # GLOBAL SEARCH FALLBACK
            # -----------------------------
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

            if not results or not isinstance(results, dict):
                yield f"I couldn't find any {search_type} matching '{query}'."
                return

            type_key = search_type + "s"
            type_data = results.get(type_key)
            
            if not type_data or not isinstance(type_data, dict):
                yield f"I couldn't find any {search_type} matching '{query}'."
                return

            items = type_data.get("items")
            if not isinstance(items, list) or not items:
                yield f"I couldn't find any {search_type} matching '{query}'."
                return

            # Find a match with a valid public URI context
            uri = None
            chosen_item = None
            for candidate in items:
                try:
                    if not candidate:
                        continue
                    c_uri = candidate.get("uri") if hasattr(candidate, "get") else getattr(candidate, "uri", None)
                    if c_uri and not ("collab" in c_uri or "mix" in c_uri and not candidate.get("images")):
                        uri = c_uri
                        chosen_item = candidate
                        break
                except Exception:
                    continue

            # Absolute fallback: if everything was filtered out, safely force take the first available item
            if not uri and items and items[0]:
                chosen_item = items[0]
                try:
                    uri = chosen_item.get("uri") if hasattr(chosen_item, "get") else getattr(chosen_item, "uri", None)
                except Exception:
                    uri = None

            if not uri or not chosen_item:
                yield f"I found results for '{query}', but Spotify restricted remote playback or returned unplayable content."
                return

            # -----------------------------
            # Playback handling (Completely Subscript-Protected)
            # -----------------------------
            if search_type == "track":
                sp.start_playback(device_id=device_id, uris=[uri])
                track_name = chosen_item.get("name", "Unknown Track") if isinstance(chosen_item, dict) else getattr(chosen_item, "name", "Unknown Track")
                yield f"Playing track: {track_name}."

            elif search_type == "artist":
                artist_id = chosen_item.get("id") if isinstance(chosen_item, dict) else getattr(chosen_item, "id", None)
                artist_name = chosen_item.get("name", "Unknown Artist") if isinstance(chosen_item, dict) else getattr(chosen_item, "name", "Unknown Artist")
                if artist_id:
                    sp.start_playback(device_id=device_id, context_uri=f"spotify:artist:{artist_id}:top-tracks")
                    yield f"Playing top tracks by {artist_name}."
                else:
                    yield "Found artist but couldn't parse a valid playback ID."

            elif search_type == "playlist":
                playlist_title = chosen_item.get("name", "Unknown Playlist") if isinstance(chosen_item, dict) else getattr(chosen_item, "name", "Unknown Playlist")
                sp.start_playback(device_id=device_id, context_uri=uri)
                yield f"Playing playlist: {playlist_title}."

        except Exception as e:
            # Extract precise traceback info
            tb = e.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            
            line_num = tb.tb_lineno
            filename = tb.tb_frame.f_code.co_filename
            
            print(f"\n[CRITICAL ERROR LOCATION] File: {filename} | Line: {line_num}")
            print(f"Error Type: {type(e).__name__} | Message: {e}\n")
            
            yield f"Spotify error at line {line_num}: {type(e).__name__}: {e}"


class SpotifyShuffleAction(Action):
    name = "spotify_shuffle"
    description = "Toggles shuffle on or off for Spotify playback."
    parameters = {
        "type": "object",
        "properties": {
            "state": {
                "type": "boolean",
                "description": "True to turn shuffle on, False to turn it off. Default is True.",
            }
        },
        "required": [],
    }

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        try:
            sp = _get_client()
            state = arguments.get("state", True)
            
            devices = sp.devices().get("devices", [])
            if not devices:
                yield "No active Spotify device found. Start playing something first."
                return
                
            device_id = _get_device_id(sp) or devices[0]["id"]
            
            sp.shuffle(state=state, device_id=device_id)
            status = "on" if state else "off"
            yield f"Shuffle turned {status}."
        except Exception as e:
            yield f"Spotify error: {e}"


class SpotifyPauseAction(Action):
    name = "spotify_pause"
    description = "Pauses Spotify playback."
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        try:
            _get_client().pause_playback()
            yield "Paused."
        except Exception as e:
            yield f"Spotify error: {e}"


class SpotifyResumeAction(Action):
    name = "spotify_resume"
    description = "Resumes Spotify playback."
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        try:
            _get_client().start_playback()
            yield "Resuming."
        except Exception as e:
            yield f"Spotify error: {e}"


class SpotifySkipAction(Action):
    name = "spotify_skip"
    description = "Skips to the next track on Spotify."
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        try:
            _get_client().next_track()
            yield "Skipped."
        except Exception as e:
            yield f"Spotify error: {e}"


class SpotifyVolumeAction(Action):
    name = "spotify_volume"
    description = (
        "Sets or adjusts the Spotify playback volume. "
        "Use for absolute levels ('set volume to 50', 'maximum volume' -> 100, 'mute' -> 0) "
        "or relative changes ('volume up', 'make it quieter')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "level": {
                "type": "integer",
                "description": "The absolute volume level from 0 to 100. For 'maximum' use 100, for 'minimum/quiet' use 10, for 'mute' use 0.",
            },
            "direction": {
                "type": "string",
                "enum": ["up", "down"],
                "description": "Relative change direction if the user didn't specify an exact number or absolute term.",
            },
        },
    }

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        try:
            sp = _get_client()
            direction = arguments.get("direction")
            level = arguments.get("level")

            # Handle relative volume changes (up / down)
            if direction:
                current = sp.current_playback()
                if not current or not current.get("device"):
                    yield "Nothing is playing right now."
                    return
                vol = current["device"].get("volume_percent", 50)
                level = min(100, vol + 20) if direction == "up" else max(0, vol - 20)

            # If the LLM completely blanked on providing a level or direction
            if level is None:
                yield "What volume level would you like?"
                return

            # Clamp the volume level between 0 and 100 just to be safe
            final_volume = max(0, min(100, int(level)))
            
            sp.volume(final_volume)
            yield f"Volume set to {final_volume}%."
        except Exception as e:
            yield f"Spotify error: {type(e).__name__}: {e}"


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
            artist = track["artists"][0]["name"]
            yield f"Playing {track['name']} by {artist}."
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