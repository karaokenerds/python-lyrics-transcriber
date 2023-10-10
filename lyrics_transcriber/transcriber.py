import os
import re
import json
import logging
import shutil
import hashlib
import subprocess
import slugify
import whisper_timestamped as whisper
import lyricsgenius
import syrics.api
from datetime import timedelta
from .tuul_utils import timing_data, subtitles


class LyricsTranscriber:
    def __init__(
        self,
        audio_filepath,
        artist=None,
        title=None,
        genius_api_token=None,
        spotify_cookie=None,
        output_dir=None,
        cache_dir="/tmp/lyrics-transcriber-cache/",
        log_level=logging.DEBUG,
        log_formatter=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.log_level = log_level
        self.log_formatter = log_formatter

        self.log_handler = logging.StreamHandler()

        if self.log_formatter is None:
            self.log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")

        self.log_handler.setFormatter(self.log_formatter)
        self.logger.addHandler(self.log_handler)

        self.logger.debug(f"LyricsTranscriber instantiating with input file: {audio_filepath}")

        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.audio_filepath = audio_filepath

        self.artist = artist
        self.title = title
        self.song_known = self.artist is not None and self.title is not None

        self.genius_api_token = os.getenv("GENIUS_API_TOKEN", default=genius_api_token)
        self.spotify_cookie = os.getenv("SPOTIFY_COOKIE_SP_DC", default=spotify_cookie)

        self.whisper_result_dict = None

        self.result_metadata = {
            "whisper_json_filepath": None,
            "genius_lyrics": None,
            "genius_lyrics_filepath": None,
            "spotify_lyrics_data_dict": None,
            "spotify_lyrics_data_filepath": None,
            "spotify_lyrics_text_filepath": None,
            "midico_lrc_filepath": None,
            "ass_subtitles_filepath": None,
            "karaoke_video_filepath": None,
            "singing_percentage": None,
            "total_singing_duration": None,
            "song_duration": None,
        }

        if self.audio_filepath is None:
            raise Exception("audio_filepath must be specified as the input source to transcribe")

        self.create_folders()

    def generate(self):
        self.logger.debug(f"audio_filepath is set: {self.audio_filepath}, beginning initial whisper transcription")

        self.result_metadata["whisper_json_filepath"] = self.get_cache_filepath(".json")
        self.whisper_result_dict = self.transcribe()

        self.calculate_singing_percentage()

        self.write_genius_lyrics_file()
        self.write_spotify_lyrics_data_file()
        self.write_spotify_lyrics_plain_text()

        # TODO: attempt to match up segments from genius lyrics with whisper segments

        self.result_metadata["midico_lrc_filepath"] = self.get_cache_filepath(".lrc")
        self.write_midico_lrc_file()

        self.result_metadata["ass_subtitles_filepath"] = self.get_cache_filepath(".ass")
        self.write_ass_file()

        self.result_metadata["karaoke_video_filepath"] = self.get_cache_filepath(".mp4")
        self.create_video()

        self.copy_files_to_output_dir()

        return self.result_metadata

    def copy_files_to_output_dir(self):
        if self.output_dir is None:
            self.output_dir = os.getcwd()

        self.logger.debug(f"copying temporary files to output dir: {self.output_dir}")

        self.result_metadata["whisper_json_filepath"] = shutil.copy(self.result_metadata["whisper_json_filepath"], self.output_dir)
        self.result_metadata["midico_lrc_filepath"] = shutil.copy(self.result_metadata["midico_lrc_filepath"], self.output_dir)

        if self.result_metadata["genius_lyrics_filepath"] is not None:
            self.result_metadata["genius_lyrics_filepath"] = shutil.copy(self.result_metadata["genius_lyrics_filepath"], self.output_dir)

        if self.result_metadata["spotify_lyrics_data_filepath"] is not None:
            self.result_metadata["spotify_lyrics_data_filepath"] = shutil.copy(
                self.result_metadata["spotify_lyrics_data_filepath"], self.output_dir
            )

        if self.result_metadata["spotify_lyrics_text_filepath"] is not None:
            self.result_metadata["spotify_lyrics_text_filepath"] = shutil.copy(
                self.result_metadata["spotify_lyrics_text_filepath"], self.output_dir
            )

        if self.result_metadata["ass_subtitles_filepath"] is not None:
            self.result_metadata["ass_subtitles_filepath"] = shutil.copy(self.result_metadata["ass_subtitles_filepath"], self.output_dir)

        if self.result_metadata["karaoke_video_filepath"] is not None:
            self.result_metadata["karaoke_video_filepath"] = shutil.copy(self.result_metadata["karaoke_video_filepath"], self.output_dir)

    def write_spotify_lyrics_data_file(self):
        if self.spotify_cookie and self.song_known:
            self.logger.debug(f"attempting spotify fetch as spotify_cookie and song name was set")
        else:
            self.logger.warning(f"skipping spotify fetch as not all spotify params were set")
            return

        spotify_lyrics_data_json_cache_filepath = os.path.join(self.cache_dir, "lyrics-" + self.get_song_slug() + "-spotify.json")

        if os.path.isfile(spotify_lyrics_data_json_cache_filepath):
            self.logger.debug(
                f"found existing file at spotify_lyrics_data_json_cache_filepath, reading: {spotify_lyrics_data_json_cache_filepath}"
            )

            with open(spotify_lyrics_data_json_cache_filepath, "r") as spotify_lyrics_data_json:
                self.result_metadata["spotify_lyrics_data_filepath"] = spotify_lyrics_data_json_cache_filepath

                spotify_lyrics_data_dict = json.load(spotify_lyrics_data_json)
                self.result_metadata["spotify_lyrics_data_dict"] = spotify_lyrics_data_dict
                return spotify_lyrics_data_dict

        self.logger.debug(
            f"no cached lyrics found at spotify_lyrics_data_json_cache_filepath: {spotify_lyrics_data_json_cache_filepath}, attempting to fetch from spotify"
        )

        spotify_lyrics_json = None

        try:
            spotify_client = syrics.api.Spotify(self.spotify_cookie)
            spotify_search_query = f"{self.title} - {self.artist}"
            spotify_search_results = spotify_client.search(spotify_search_query, type="track", limit=5)

            spotify_top_result = spotify_search_results["tracks"]["items"][0]
            self.logger.debug(
                f"spotify_top_result: {spotify_top_result['artists'][0]['name']} - {spotify_top_result['name']} ({spotify_top_result['external_urls']['spotify']})"
            )

            spotify_lyrics_dict = spotify_client.get_lyrics(spotify_top_result["id"])
            spotify_lyrics_json = json.dumps(spotify_lyrics_dict, indent=4)

            self.logger.debug(
                f"writing lyrics data JSON to spotify_lyrics_data_json_cache_filepath: {spotify_lyrics_data_json_cache_filepath}"
            )
            with open(spotify_lyrics_data_json_cache_filepath, "w") as f:
                f.write(spotify_lyrics_json)
        except Exception as e:
            self.logger.warn(f"caught exception while attempting to fetch from spotify: ", e)

        self.result_metadata["spotify_lyrics_data_filepath"] = spotify_lyrics_data_json_cache_filepath
        self.result_metadata["spotify_lyrics_data_dict"] = spotify_lyrics_dict
        return spotify_lyrics_dict

    def write_spotify_lyrics_plain_text(self):
        if self.result_metadata["spotify_lyrics_data_dict"]:
            self.logger.debug(f"spotify_lyrics data found, checking/writing plain text lyrics file")

            spotify_lyrics_text_filepath = os.path.join(self.cache_dir, "lyrics-" + self.get_song_slug() + "-spotify.txt")
            self.result_metadata["spotify_lyrics_text_filepath"] = spotify_lyrics_text_filepath

            lines = self.result_metadata["spotify_lyrics_data_dict"]["lyrics"]["lines"]

            self.logger.debug(f"writing lyrics plain text to spotify_lyrics_text_filepath: {spotify_lyrics_text_filepath}")
            with open(spotify_lyrics_text_filepath, "w") as f:
                for line in lines:
                    f.write(line["words"] + "\n")

    def write_genius_lyrics_file(self):
        if self.genius_api_token and self.song_known:
            self.logger.debug(f"attempting genius fetch as genius_api_token and song name was set")
        else:
            self.logger.warning(f"skipping genius fetch as not all genius params were set")
            return

        genius_lyrics_cache_filepath = os.path.join(self.cache_dir, "lyrics-" + self.get_song_slug() + "-genius.txt")

        if os.path.isfile(genius_lyrics_cache_filepath):
            self.logger.debug(f"found existing file at genius_lyrics_cache_filepath, reading: {genius_lyrics_cache_filepath}")

            with open(genius_lyrics_cache_filepath, "r") as cached_lyrics:
                self.result_metadata["genius_lyrics_filepath"] = genius_lyrics_cache_filepath
                self.result_metadata["genius_lyrics"] = cached_lyrics
                return cached_lyrics

        self.logger.debug(f"no cached lyrics found at genius_lyrics_cache_filepath: {genius_lyrics_cache_filepath}, fetching from Genius")
        genius = lyricsgenius.Genius(self.genius_api_token, verbose=(self.log_level == logging.DEBUG))

        song = genius.search_song(self.title, self.artist)
        if song is None:
            self.logger.warning(f'Could not find lyrics on Genius for "{self.title}" by {self.artist}')
            return
        lyrics = self.clean_genius_lyrics(song.lyrics)

        self.logger.debug(f"writing clean lyrics to genius_lyrics_cache_filepath: {genius_lyrics_cache_filepath}")
        with open(genius_lyrics_cache_filepath, "w") as f:
            f.write(lyrics)

        self.result_metadata["genius_lyrics_filepath"] = genius_lyrics_cache_filepath
        self.result_metadata["genius_lyrics"] = lyrics
        return lyrics

    def clean_genius_lyrics(self, lyrics):
        lyrics = lyrics.replace("\\n", "\n")
        lyrics = re.sub(r"You might also like", "", lyrics)
        # Remove the song name and word "Lyrics" if this has a non-newline char at the start
        lyrics = re.sub(r".*?Lyrics([A-Z])", r"\1", lyrics)
        lyrics = re.sub(r"[0-9]+Embed$", "", lyrics)  # Remove the word "Embed" at end of line with preceding numbers if found
        lyrics = re.sub(r"(\S)Embed$", r"\1", lyrics)  # Remove the word "Embed" if it has been tacked onto a word at the end of a line
        lyrics = re.sub(r"^Embed$", r"", lyrics)  # Remove the word "Embed" if it has been tacked onto a word at the end of a line
        lyrics = re.sub(r".*?\[.*?\].*?", "", lyrics)  # Remove lines containing square brackets
        # add any additional cleaning rules here
        return lyrics

    def calculate_singing_percentage(self):
        # Calculate total seconds of singing using timings from whisper transcription results
        total_singing_duration = sum(segment["end"] - segment["start"] for segment in self.whisper_result_dict["segments"])

        self.logger.debug(f"calculated total_singing_duration: {int(total_singing_duration)} seconds, now running ffprobe")

        # Calculate total song duration using ffprobe
        duration_command = [
            "ffprobe",
            "-i",
            self.audio_filepath,
            "-show_entries",
            "format=duration",
            "-v",
            "quiet",
            "-of",
            "csv=%s" % ("p=0"),
        ]
        duration_output = subprocess.check_output(duration_command, universal_newlines=True)
        song_duration = float(duration_output)

        # Calculate singing percentage
        singing_percentage = int((total_singing_duration / song_duration) * 100)

        self.result_metadata["singing_percentage"] = singing_percentage
        self.result_metadata["total_singing_duration"] = total_singing_duration
        self.result_metadata["song_duration"] = song_duration

        return singing_percentage, total_singing_duration, song_duration

    # Loops through lyrics segments (typically sentences) from whisper_timestamps JSON output,
    # then loops over each word and writes all words with MidiCo segment start/end formatting
    # and word-level timestamps to a MidiCo-compatible LRC file
    def write_midico_lrc_file(self):
        lrc_filename = self.result_metadata["midico_lrc_filepath"]
        self.logger.debug(f"writing midico formatted word timestamps to LRC file: {lrc_filename}")
        with open(lrc_filename, "w") as f:
            f.write("[re:MidiCo]\n")
            for segment in self.whisper_result_dict["segments"]:
                for i, word in enumerate(segment["words"]):
                    start_time = self.format_time_lrc(word["start"])
                    if i != len(segment["words"]) - 1:
                        word["text"] += " "
                    line = "[{}]1:{}{}\n".format(start_time, "/" if i == 0 else "", word["text"])
                    f.write(line)

    def write_ass_file(self):
        ass_filepath = self.result_metadata["ass_subtitles_filepath"]
        self.logger.debug(f"writing ASS formatted subtitle file: {ass_filepath}")

        spotify_lines = self.result_metadata["spotify_lyrics_data_dict"]["lyrics"]["lines"]
        simple_text_lines = []
        events_tuples = []

        for line in spotify_lines:
            simple_text_lines.append(line["words"])
            startTimeDelta = timedelta(milliseconds=int(line["startTimeMs"]))
            events_tuples.append((startTimeDelta, timing_data.LyricMarker.SEGMENT_START))

        intial_screens = subtitles.create_screens(self.logger, simple_text_lines, events_tuples)
        screens = subtitles.set_segment_end_times(intial_screens, int(self.result_metadata["song_duration"]))
        screens = subtitles.set_screen_start_times(screens)
        lyric_subtitles_ass = subtitles.create_subtitles(
            screens,
            {
                "FontName": "Arial Narrow",
                "FontSize": 20,
                "PrimaryColor": (255, 0, 255, 255),
                "SecondaryColor": (0, 255, 255, 255),
            },
        )

        lyric_subtitles_ass.write(ass_filepath)

    def create_video(self):
        self.logger.debug(f"create_video attempting to generate video file: {self.result_metadata['karaoke_video_filepath']}")

        audio_delay = 0
        audio_delay_ms = int(audio_delay * 1000)  # milliseconds

        video_metadata = []
        if self.artist:
            video_metadata.append("-metadata")
            video_metadata.append(f"artist={self.artist}")
        if self.title:
            video_metadata.append("-metadata")
            video_metadata.append(f"title={self.title}")

        ffmpeg_cmd = [
            "ffmpeg",
            # Describe a video stream that is a black background
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1280x720:r=20",
            # Use accompaniment track as audio
            "-i",
            self.audio_filepath,
            # Set audio delay if needed
            # https://ffmpeg.org/ffmpeg-filters.html#adelay
            "-af",
            f"adelay=delays={audio_delay_ms}:all=1",
            # Re-encode audio as mp3
            "-c:a",
            "libmp3lame",
            # Add subtitles
            "-vf",
            "ass=" + self.result_metadata["ass_subtitles_filepath"],
            # End encoding after the shortest stream
            "-shortest",
            # Overwrite files without asking
            "-y",
            *video_metadata,
            # Output path of video
            self.result_metadata["karaoke_video_filepath"],
        ]

        self.logger.debug(f"running ffmpeg command to generate video: {ffmpeg_cmd}")
        ffmpeg_output = subprocess.check_output(ffmpeg_cmd, universal_newlines=True)
        return ffmpeg_output

    def format_time_lrc(self, duration):
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        milliseconds = int((duration % 1) * 1000)
        formatted_time = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        return formatted_time

    def transcribe(self):
        whisper_cache_filepath = self.result_metadata["whisper_json_filepath"]
        if os.path.isfile(whisper_cache_filepath):
            self.logger.debug(f"transcribe found existing file at whisper_cache_filepath, reading: {whisper_cache_filepath}")
            with open(whisper_cache_filepath, "r") as cache_file:
                return json.load(cache_file)

        self.logger.debug(f"no cached transcription file found, running whisper transcribe")
        audio = whisper.load_audio(self.audio_filepath)
        model = whisper.load_model("medium.en", device="cpu")
        result = whisper.transcribe(model, audio, language="en")

        self.logger.debug(f"whisper transcription complete, writing JSON to cache file: {whisper_cache_filepath}")
        with open(whisper_cache_filepath, "w") as cache_file:
            json.dump(result, cache_file, indent=4)

        return result

    def get_cache_filepath(self, extension):
        filename = os.path.split(self.audio_filepath)[1]
        filename_slug = slugify.slugify(filename, lowercase=False)
        hash_value = self.get_file_hash(self.audio_filepath)
        cache_filepath = os.path.join(self.cache_dir, filename_slug + "_" + hash_value + extension)
        self.logger.debug(f"get_cache_filepath returning cache_filepath: {cache_filepath}")
        return cache_filepath

    def get_song_slug(self):
        artist_slug = slugify.slugify(self.artist, lowercase=False)
        title_slug = slugify.slugify(self.title, lowercase=False)
        return artist_slug + "-" + title_slug

    def get_file_hash(self, filepath):
        return hashlib.md5(open(filepath, "rb").read()).hexdigest()

    def create_folders(self):
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
