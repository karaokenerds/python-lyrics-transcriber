import os
import sys
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
from .utils import subtitles
from typing import List, Optional
from openai import OpenAI


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
        transcription_model="medium",
        llm_model="gpt-4o",
        llm_prompt_matching=f"{os.path.dirname(__file__)}/llm_prompts/llm_prompt_lyrics_matching_andrew_handwritten_20231118.txt",
        llm_prompt_correction=f"{os.path.dirname(__file__)}/llm_prompts/llm_prompt_lyrics_correction_andrew_handwritten_20231118.txt",
        render_video=False,
        video_resolution="360p",
        video_background_image=None,
        video_background_color="black",
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

        self.transcription_model = transcription_model
        self.llm_model = llm_model
        self.llm_prompt_matching = llm_prompt_matching
        self.llm_prompt_correction = llm_prompt_correction

        self.openai_client = OpenAI()

        # Uncomment for local models e.g. with ollama
        # self.openai_client = OpenAI(
        #     base_url="http://localhost:11434/v1",
        #     api_key="ollama",
        # )

        self.openai_client.log = self.log_level

        self.render_video = render_video
        self.video_resolution = video_resolution
        self.video_background_image = video_background_image
        self.video_background_color = video_background_color

        match video_resolution:
            case "4k":
                self.video_resolution_num = (3840, 2160)
                self.font_size = 250
                self.line_height = 250
            case "1080p":
                self.video_resolution_num = (1920, 1080)
                self.font_size = 120
                self.line_height = 120
            case "720p":
                self.video_resolution_num = (1280, 720)
                self.font_size = 100
                self.line_height = 100
            case "360p":
                self.video_resolution_num = (640, 360)
                self.font_size = 50
                self.line_height = 50
            case _:
                raise ValueError("Invalid video_resolution value. Must be one of: 4k, 1080p, 720p, 360p")

        # If a video background is provided, validate file exists
        if self.video_background_image is not None:
            if os.path.isfile(self.video_background_image):
                self.logger.debug(f"video_background is valid file path: {self.video_background_image}")
            else:
                raise FileNotFoundError(f"video_background is not a valid file path: {self.video_background_image}")

        self.outputs = {
            "transcription_data_dict": None,
            "transcription_data_filepath": None,
            "transcribed_lyrics_text": None,
            "transcribed_lyrics_text_filepath": None,
            "genius_lyrics_text": None,
            "genius_lyrics_filepath": None,
            "spotify_lyrics_data_dict": None,
            "spotify_lyrics_data_filepath": None,
            "spotify_lyrics_text_filepath": None,
            "llm_token_usage": {"input": 0, "output": 0},
            "llm_costs_usd": {"input": 0.0, "output": 0.0, "total": 0.0},
            "llm_transcript": None,
            "llm_transcript_filepath": None,
            "corrected_lyrics_text": None,
            "corrected_lyrics_text_filepath": None,
            "midico_lrc_filepath": None,
            "ass_subtitles_filepath": None,
            "karaoke_video_filepath": None,
            "singing_percentage": None,
            "total_singing_duration": None,
            "song_duration": None,
            "output_dir": None,
        }

        if self.audio_filepath is None:
            raise Exception("audio_filepath must be specified as the input source to transcribe")

        if not self.song_known:
            raise Exception("cannot correct song lyrics without artist and title to fetch lyrics")

        self.create_folders()

    def generate(self):
        self.logger.debug(f"audio_filepath is set: {self.audio_filepath}, beginning initial whisper transcription")

        self.transcribe()
        self.write_transcribed_lyrics_plain_text()

        self.write_genius_lyrics_file()
        self.write_spotify_lyrics_data_file()
        self.write_spotify_lyrics_plain_text()

        self.validate_lyrics_match_song()

        self.write_corrected_lyrics_data_file()
        self.write_corrected_lyrics_plain_text()

        self.calculate_singing_percentage()

        self.write_midico_lrc_file()
        self.write_ass_file()

        if self.render_video:
            self.outputs["karaoke_video_filepath"] = self.get_cache_filepath(".mp4")
            self.create_video()

        self.copy_files_to_output_dir()
        self.calculate_llm_costs()

        self.openai_client.close()

        return self.outputs

    def copy_files_to_output_dir(self):
        if self.output_dir is None:
            self.output_dir = os.getcwd()

        self.logger.debug(f"copying temporary files to output dir: {self.output_dir}")

        for key in self.outputs:
            if key.endswith("_filepath"):
                if self.outputs[key] and os.path.isfile(self.outputs[key]):
                    shutil.copy(self.outputs[key], self.output_dir)

        self.outputs["output_dir"] = self.output_dir

    def validate_lyrics_match_song(self):
        at_least_one_online_lyrics_validated = False

        with open(self.llm_prompt_matching, "r") as file:
            llm_matching_instructions = file.read()

        for online_lyrics_source in ["genius", "spotify"]:
            self.logger.debug(f"validating transcribed lyrics match lyrics from {online_lyrics_source}")

            online_lyrics_text_key = f"{online_lyrics_source}_lyrics_text"
            online_lyrics_filepath_key = f"{online_lyrics_source}_lyrics_filepath"

            if online_lyrics_text_key not in self.outputs:
                continue

            data_input_str = (
                f'Data input 1:\n{self.outputs["transcribed_lyrics_text"]}\nData input 2:\n{self.outputs[online_lyrics_text_key]}\n'
            )

            # self.logger.debug(f"system_prompt:\n{system_prompt}\ndata_input_str:\n{data_input_str}")

            self.logger.debug(f"making API call to LLM model {self.llm_model} to validate {online_lyrics_source} lyrics match")
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": llm_matching_instructions}, {"role": "user", "content": data_input_str}],
            )

            message = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            self.outputs["llm_token_usage"]["input"] += response.usage.prompt_tokens
            self.outputs["llm_token_usage"]["output"] += response.usage.completion_tokens

            # self.logger.debug(f"LLM API response finish_reason: {finish_reason} message: \n{message}")

            if finish_reason == "stop":
                if message == "Yes":
                    self.logger.info(f"{online_lyrics_source} lyrics successfully validated to match transcription")
                    at_least_one_online_lyrics_validated = True
                elif message == "No":
                    self.logger.warning(f"{online_lyrics_source} lyrics do not match transcription, deleting that source from outputs")
                    self.outputs[online_lyrics_text_key] = None
                    self.outputs[online_lyrics_filepath_key] = None
                else:
                    self.logger.error(f"Unexpected response from LLM: {message}")
            else:
                self.logger.warning(f"OpenAI API call did not finish successfully, finish_reason: {finish_reason}")

        self.logger.info(
            f"Completed validation of transcription using online lyrics sources. Match found: {at_least_one_online_lyrics_validated}"
        )

        if not at_least_one_online_lyrics_validated:
            self.logger.error(
                f"Lyrics from Genius and Spotify did not match the transcription. Please check artist and title are set correctly."
            )
            raise Exception("Cannot proceed without internet lyrics to validate / correct transcription")

    def write_corrected_lyrics_data_file(self):
        self.logger.debug("write_corrected_lyrics_data_file initiating OpenAI client")

        corrected_lyrics_data_json_cache_filepath = os.path.join(self.cache_dir, "lyrics-" + self.get_song_slug() + "-corrected.json")

        if os.path.isfile(corrected_lyrics_data_json_cache_filepath):
            self.logger.debug(
                f"found existing file at corrected_lyrics_data_json_cache_filepath, reading: {corrected_lyrics_data_json_cache_filepath}"
            )

            with open(corrected_lyrics_data_json_cache_filepath, "r") as corrected_lyrics_data_json:
                self.outputs["corrected_lyrics_data_filepath"] = corrected_lyrics_data_json_cache_filepath

                corrected_lyrics_data_dict = json.load(corrected_lyrics_data_json)
                self.outputs["corrected_lyrics_data_dict"] = corrected_lyrics_data_dict
                return

        self.logger.debug(
            f"no cached lyrics found at corrected_lyrics_data_json_cache_filepath: {corrected_lyrics_data_json_cache_filepath}, attempting to run correction using LLM"
        )

        corrected_lyrics_dict = {"segments": []}

        with open(self.llm_prompt_correction, "r") as file:
            system_prompt_template = file.read()

        reference_lyrics = self.outputs["genius_lyrics_text"] or self.outputs["spotify_lyrics_text"]
        system_prompt = system_prompt_template.replace("{{reference_lyrics}}", reference_lyrics)

        # TODO: Test if results are cleaner when using the vocal file from a background vocal audio separation model
        # TODO: Record more info about the correction process (e.g before/after diffs for each segment) to a file for debugging
        # TODO: Possibly add a step after segment-based correct to get the LLM to self-analyse the diff

        self.outputs["llm_transcript"] = ""
        self.outputs["llm_transcript_filepath"] = os.path.join(
            self.cache_dir, "lyrics-" + self.get_song_slug() + "-llm-correction-transcript.txt"
        )

        total_segments = len(self.outputs["transcription_data_dict"]["segments"])
        self.logger.info(f"Beginning correction using LLM, total segments: {total_segments}")

        with open(self.outputs["llm_transcript_filepath"], "a", buffering=1, encoding="utf-8") as llm_transcript_file:
            self.logger.debug(f"writing LLM chat instructions: {self.outputs['llm_transcript_filepath']}")

            llm_transcript_header = f"--- SYSTEM instructions passed in for all segments ---:\n\n{system_prompt}\n"
            self.outputs["llm_transcript"] += llm_transcript_header
            llm_transcript_file.write(llm_transcript_header)

            for segment in self.outputs["transcription_data_dict"]["segments"]:
                # # Don't waste OpenAI dollars when testing!
                # if segment["id"] > 10:
                #     continue
                # if segment["id"] < 20 or segment["id"] > 24:
                #     continue

                llm_transcript_segment = ""
                segment_input = json.dumps(
                    {
                        "id": segment["id"],
                        "start": segment["start"],
                        "end": segment["end"],
                        "confidence": segment["confidence"],
                        "text": segment["text"],
                        "words": segment["words"],
                    }
                )

                previous_two_corrected_lines = ""
                upcoming_two_uncorrected_lines = ""

                for previous_segment in corrected_lyrics_dict["segments"]:
                    if previous_segment["id"] in (segment["id"] - 2, segment["id"] - 1):
                        previous_two_corrected_lines += previous_segment["text"].strip() + "\n"

                for next_segment in self.outputs["transcription_data_dict"]["segments"]:
                    if next_segment["id"] in (segment["id"] + 1, segment["id"] + 2):
                        upcoming_two_uncorrected_lines += next_segment["text"].strip() + "\n"

                llm_transcript_segment += f"--- Segment {segment['id']} / {total_segments} ---\n"
                llm_transcript_segment += f"Previous two corrected lines:\n\n{previous_two_corrected_lines}\nUpcoming two uncorrected lines:\n\n{upcoming_two_uncorrected_lines}\nData input:\n\n{segment_input}\n"

                # fmt: off
                segment_prompt = system_prompt_template.replace(
                    "{{previous_two_corrected_lines}}", previous_two_corrected_lines
                ).replace(
                    "{{upcoming_two_uncorrected_lines}}", upcoming_two_uncorrected_lines
                ).replace(
                    "{{segment_input}}", segment_input
                )

                self.logger.info(
                    f'Calling completion model {self.llm_model} with instructions and data input for segment {segment["id"]} / {total_segments}:'
                )

                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    response_format={"type": "json_object"},
                    seed=10,
                    temperature=0.4,
                    messages=[
                        {
                            "role": "user", 
                            "content": segment_prompt
                        }
                    ],
                )
                # fmt: on

                message = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                llm_transcript_segment += f"\n--- RESPONSE for segment {segment['id']} ---:\n\n"
                llm_transcript_segment += message
                llm_transcript_segment += f"\n--- END segment {segment['id']} / {total_segments} ---:\n\n"

                self.logger.debug(f"writing LLM chat transcript for segment to: {self.outputs['llm_transcript_filepath']}")
                llm_transcript_file.write(llm_transcript_segment)
                self.outputs["llm_transcript"] += llm_transcript_segment

                self.outputs["llm_token_usage"]["input"] += response.usage.prompt_tokens
                self.outputs["llm_token_usage"]["output"] += response.usage.completion_tokens

                # self.logger.debug(f"response finish_reason: {finish_reason} message: \n{message}")

                if finish_reason == "stop":
                    try:
                        corrected_segment_dict = json.loads(message)
                        corrected_lyrics_dict["segments"].append(corrected_segment_dict)
                        self.logger.info("Successfully parsed response from GPT as JSON and appended to corrected_lyrics_dict.segments")
                    except json.JSONDecodeError as e:
                        raise Exception("Failed to parse response from GPT as JSON") from e
                else:
                    self.logger.warning(f"OpenAI API call did not finish successfully, finish_reason: {finish_reason}")

            self.logger.info(f'Successfully processed correction for all {len(corrected_lyrics_dict["segments"])} lyrics segments')

            self.logger.debug(f"writing corrected lyrics data JSON filepath: {corrected_lyrics_data_json_cache_filepath}")
            with open(corrected_lyrics_data_json_cache_filepath, "w", encoding="utf-8") as corrected_lyrics_data_json_cache_file:
                corrected_lyrics_data_json_cache_file.write(json.dumps(corrected_lyrics_dict, indent=4))

        self.outputs["corrected_lyrics_data_filepath"] = corrected_lyrics_data_json_cache_filepath
        self.outputs["corrected_lyrics_data_dict"] = corrected_lyrics_dict

    def calculate_llm_costs(self):
        price_dollars_per_1000_tokens = {
            "gpt-3.5-turbo-1106": {
                "input": 0.0010,
                "output": 0.0020,
            },
            "gpt-4-1106-preview": {
                "input": 0.01,
                "output": 0.03,
            },
        }

        input_price = price_dollars_per_1000_tokens.get(self.llm_model, {"input": 0, "output": 0})["input"]
        output_price = price_dollars_per_1000_tokens.get(self.llm_model, {"input": 0, "output": 0})["output"]

        input_cost = input_price * (self.outputs["llm_token_usage"]["input"] / 1000)
        output_cost = output_price * (self.outputs["llm_token_usage"]["output"] / 1000)

        self.outputs["llm_costs_usd"]["input"] = round(input_cost, 3)
        self.outputs["llm_costs_usd"]["output"] = round(output_cost, 3)
        self.outputs["llm_costs_usd"]["total"] = round(input_cost + output_cost, 3)

    def write_corrected_lyrics_plain_text(self):
        if self.outputs["corrected_lyrics_data_dict"]:
            self.logger.debug(f"corrected_lyrics_data_dict exists, writing plain text lyrics file")

            corrected_lyrics_text_filepath = os.path.join(self.cache_dir, "lyrics-" + self.get_song_slug() + "-corrected.txt")
            self.outputs["corrected_lyrics_text_filepath"] = corrected_lyrics_text_filepath

            self.outputs["corrected_lyrics_text"] = ""

            self.logger.debug(f"writing lyrics plain text to corrected_lyrics_text_filepath: {corrected_lyrics_text_filepath}")
            with open(corrected_lyrics_text_filepath, "w", encoding="utf-8") as f:
                for corrected_segment in self.outputs["corrected_lyrics_data_dict"]["segments"]:
                    self.outputs["corrected_lyrics_text"] += corrected_segment["text"].strip() + "\n"
                    f.write(corrected_segment["text".strip()] + "\n")

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
                spotify_lyrics_data_dict = json.load(spotify_lyrics_data_json)
                self.outputs["spotify_lyrics_data_filepath"] = spotify_lyrics_data_json_cache_filepath
                self.outputs["spotify_lyrics_data_dict"] = spotify_lyrics_data_dict
                return

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
            with open(spotify_lyrics_data_json_cache_filepath, "w", encoding="utf-8") as f:
                f.write(spotify_lyrics_json)
        except Exception as e:
            self.logger.warn(f"caught exception while attempting to fetch from spotify: ", e)

        self.outputs["spotify_lyrics_data_filepath"] = spotify_lyrics_data_json_cache_filepath
        self.outputs["spotify_lyrics_data_dict"] = spotify_lyrics_dict

    def write_spotify_lyrics_plain_text(self):
        if self.outputs["spotify_lyrics_data_dict"]:
            self.logger.debug(f"spotify_lyrics data found, checking/writing plain text lyrics file")

            spotify_lyrics_text_filepath = os.path.join(self.cache_dir, "lyrics-" + self.get_song_slug() + "-spotify.txt")
            self.outputs["spotify_lyrics_text_filepath"] = spotify_lyrics_text_filepath

            lines = self.outputs["spotify_lyrics_data_dict"]["lyrics"]["lines"]

            self.outputs["spotify_lyrics_text"] = ""

            self.logger.debug(f"writing lyrics plain text to spotify_lyrics_text_filepath: {spotify_lyrics_text_filepath}")
            with open(spotify_lyrics_text_filepath, "w", encoding="utf-8") as f:
                for line in lines:
                    self.outputs["spotify_lyrics_text"] += line["words"].strip() + "\n"
                    f.write(line["words"].strip() + "\n")

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
                self.outputs["genius_lyrics_filepath"] = genius_lyrics_cache_filepath
                self.outputs["genius_lyrics_text"] = cached_lyrics.read()
                return

        self.logger.debug(f"no cached lyrics found at genius_lyrics_cache_filepath: {genius_lyrics_cache_filepath}, fetching from Genius")
        genius = lyricsgenius.Genius(self.genius_api_token, verbose=(self.log_level == logging.DEBUG))

        song = genius.search_song(self.title, self.artist)
        if song is None:
            self.logger.warning(f'Could not find lyrics on Genius for "{self.title}" by {self.artist}')
            return
        lyrics = self.clean_genius_lyrics(song.lyrics)

        self.logger.debug(f"writing clean lyrics to genius_lyrics_cache_filepath: {genius_lyrics_cache_filepath}")
        with open(genius_lyrics_cache_filepath, "w", encoding="utf-8") as f:
            f.write(lyrics)

        self.outputs["genius_lyrics_filepath"] = genius_lyrics_cache_filepath
        self.outputs["genius_lyrics_text"] = lyrics

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
        total_singing_duration = sum(segment["end"] - segment["start"] for segment in self.outputs["transcription_data_dict"]["segments"])

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

        self.outputs["singing_percentage"] = singing_percentage
        self.outputs["total_singing_duration"] = total_singing_duration
        self.outputs["song_duration"] = song_duration

    # Loops through lyrics segments (typically sentences) from whisper_timestamps JSON output,
    # then loops over each word and writes all words with MidiCo segment start/end formatting
    # and word-level timestamps to a MidiCo-compatible LRC file
    def write_midico_lrc_file(self):
        self.outputs["midico_lrc_filepath"] = self.get_cache_filepath(".lrc")

        lrc_filename = self.outputs["midico_lrc_filepath"]
        self.logger.debug(f"writing midico formatted word timestamps to LRC file: {lrc_filename}")
        with open(lrc_filename, "w", encoding="utf-8") as f:
            f.write("[re:MidiCo]\n")
            for segment in self.outputs["corrected_lyrics_data_dict"]["segments"]:
                for i, word in enumerate(segment["words"]):
                    start_time = self.format_time_lrc(word["start"])
                    if i != len(segment["words"]) - 1:
                        word["text"] += " "
                    line = "[{}]1:{}{}\n".format(start_time, "/" if i == 0 else "", word["text"])
                    f.write(line)

    def create_screens(self):
        self.logger.debug(f"create_screens beginning generation of screens from whisper results")
        screens: List[subtitles.LyricsScreen] = []
        line: Optional[subtitles.LyricsLine] = None
        screen: Optional[subtitles.LyricsScreen] = None

        lines_in_current_screen = 0
        for segment in self.outputs["corrected_lyrics_data_dict"]["segments"]:
            self.logger.debug(f"lines_in_current_screen: {lines_in_current_screen} segment: {segment['text']}")
            if screen is None:
                self.logger.debug(f"screen is none, creating new LyricsScreen")
                screen = subtitles.LyricsScreen()
                screen.video_size = self.video_resolution_num
                screen.line_height = self.line_height
            if line is None:
                self.logger.debug(f"line is none, creating new LyricsLine")
                line = subtitles.LyricsLine()

            num_words_in_segment = len(segment["words"])
            for word_index, word in enumerate(segment["words"]):
                segment = subtitles.LyricSegment(
                    text=word["text"], ts=timedelta(seconds=word["start"]), end_ts=timedelta(seconds=word["end"])
                )
                line.segments.append(segment)

                # If word is last in the line, add line to screen and start new line
                # Before looping to the next word
                if word_index == num_words_in_segment - 1:
                    self.logger.debug(f"word_index is last in segment, adding line to screen and starting new line")
                    screen.lines.append(line)
                    lines_in_current_screen += 1
                    line = None

            # If current screen has 2 lines already, add screen to list and start new screen
            # Before looping to the next line
            if lines_in_current_screen == 2:
                self.logger.debug(f"lines_in_current_screen is 2, adding screen to list and starting new screen")
                screens.append(screen)
                screen = None
                lines_in_current_screen = 0

        if line is not None:
            screen.lines.append(line)  # type: ignore[union-attr]
        if screen is not None and len(screen.lines) > 0:
            screens.append(screen)  # type: ignore[arg-type]

        return screens

    def write_ass_file(self):
        self.outputs["ass_subtitles_filepath"] = self.get_cache_filepath(".ass")

        ass_filepath = self.outputs["ass_subtitles_filepath"]
        self.logger.debug(f"writing ASS formatted subtitle file: {ass_filepath}")

        intial_screens = self.create_screens()
        screens = subtitles.set_segment_end_times(intial_screens, int(self.outputs["song_duration"]))
        screens = subtitles.set_screen_start_times(screens)
        lyric_subtitles_ass = subtitles.create_styled_subtitles(screens, self.video_resolution_num, self.font_size)
        lyric_subtitles_ass.write(ass_filepath)

    def resize_background_image(self):
        self.logger.debug(
            f"resize_background_image attempting to resize background image: {self.video_background_image} to resolution: {self.video_resolution}"
        )
        background_image_resized = self.get_cache_filepath(f"-{self.video_resolution}.png")

        if os.path.isfile(background_image_resized):
            self.logger.debug(
                f"resize_background_image found existing resized background image, skipping resize: {background_image_resized}"
            )
            return background_image_resized

        resize_command = ["ffmpeg", "-i", self.video_background_image]
        resize_command += ["-vf", f"scale={self.video_resolution_num[0]}x{self.video_resolution_num[1]}"]

        resize_command += [background_image_resized]
        subprocess.check_output(resize_command, universal_newlines=True)

        if not os.path.isfile(background_image_resized):
            raise FileNotFoundError(
                f"background_image_resized was not a valid file after running ffmpeg to resize: {background_image_resized}"
            )

        return background_image_resized

    def create_video(self):
        self.logger.debug(f"create_video attempting to generate video file: {self.outputs['karaoke_video_filepath']}")

        audio_delay = 0
        audio_delay_ms = int(audio_delay * 1000)  # milliseconds

        video_metadata = []
        if self.artist:
            video_metadata.append("-metadata")
            video_metadata.append(f"artist={self.artist}")
        if self.title:
            video_metadata.append("-metadata")
            video_metadata.append(f"title={self.title}")

        # fmt: off
        ffmpeg_cmd = [
            "ffmpeg",
            "-r", "30", # Set frame rate to 30 fps
        ]

        if self.video_background_image:
            self.logger.debug(f"background image set: {self.video_background_image}, resizing to resolution: {self.video_resolution}")

            background_image_resized = self.resize_background_image()

            ffmpeg_cmd += [
                # Use provided image as background
                "-loop", "1",  # Loop the image
                "-i", background_image_resized,  # Input image file
            ]

        else:
            self.logger.debug(f"background not set, using solid {self.video_background_color} background with resolution: {self.video_resolution}")
            ffmpeg_cmd += ["-f", "lavfi"]
            ffmpeg_cmd += ["-i", f"color=c={self.video_background_color}:s={self.video_resolution_num[0]}x{self.video_resolution_num[1]}:r=30"]


        # Check for hardware acclerated h.264 encoding and use if available
        video_codec = "libx264"
        ffmpeg_codes = subprocess.getoutput("ffmpeg -codecs")

        if "h264_videotoolbox" in ffmpeg_codes:
            video_codec = "h264_videotoolbox"
            self.logger.info(f"video codec set to hardware accelerated h264_videotoolbox")
        elif "h264_qsv" in ffmpeg_codes:
            video_codec = "h264_qsv"
            self.logger.info(f"video codec set to hardware accelerated h264_qsv")

        ffmpeg_cmd += [
            # Use accompaniment track as audio
            "-i", self.audio_filepath,
            # Set audio delay if needed
            # https://ffmpeg.org/ffmpeg-filters.html#adelay
            # "-af",
            # f"adelay=delays={audio_delay_ms}:all=1",
            # Re-encode audio as mp3
            "-c:a", "aac",
            # Add subtitles
            "-vf", "ass=" + self.outputs["ass_subtitles_filepath"],
            # Encode as H264 using hardware acceleration if available
            "-c:v", video_codec,
            # Increase output video quality
            "-preset", "slow",  # Use a slower preset for better compression efficiency
            # "-crf", "1",  # Lower CRF for higher quality. Adjust as needed, lower is better quality
            "-b:v", "5000k",  # Set the video bitrate, for example, 5000 kbps
            "-minrate", "5000k",  # Minimum bitrate
            "-maxrate", "20000k",  # Maximum bitrate
            "-bufsize", "10000k",  # Set the buffer size, typically 2x maxrate
            # End encoding after the shortest stream
            "-shortest",
            # Overwrite files without asking
            "-y",
            # Only encode the first 30 seconds (for testing, fast iteration when editing this)
            # "-t", "30",
            *video_metadata,
            # Output path of video
            self.outputs["karaoke_video_filepath"],
        ]
        # fmt: on

        self.logger.debug(f"running ffmpeg command to generate video: {' '.join(ffmpeg_cmd)}")
        ffmpeg_output = subprocess.check_output(ffmpeg_cmd, universal_newlines=True)
        return ffmpeg_output

    def format_time_lrc(self, duration):
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        milliseconds = int((duration % 1) * 1000)
        formatted_time = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        return formatted_time

    def write_transcribed_lyrics_plain_text(self):
        if self.outputs["transcription_data_dict"]:
            transcribed_lyrics_text_filepath = os.path.join(self.cache_dir, "lyrics-" + self.get_song_slug() + "-transcribed.txt")
            self.outputs["transcribed_lyrics_text_filepath"] = transcribed_lyrics_text_filepath

            self.outputs["transcribed_lyrics_text"] = ""

            self.logger.debug(f"writing lyrics plain text to transcribed_lyrics_text_filepath: {transcribed_lyrics_text_filepath}")
            with open(transcribed_lyrics_text_filepath, "w", encoding="utf-8") as f:
                for segment in self.outputs["transcription_data_dict"]["segments"]:
                    self.outputs["transcribed_lyrics_text"] += segment["text"] + "\n"
                    f.write(segment["text"].strip() + "\n")
        else:
            raise Exception("Cannot write transcribed lyrics plain text as transcription_data_dict is not set")

    def transcribe(self):
        self.outputs["transcription_data_filepath"] = self.get_cache_filepath(".json")

        whisper_cache_filepath = self.outputs["transcription_data_filepath"]
        if os.path.isfile(whisper_cache_filepath):
            self.logger.debug(f"transcribe found existing file at whisper_cache_filepath, reading: {whisper_cache_filepath}")
            with open(whisper_cache_filepath, "r") as cache_file:
                self.outputs["transcription_data_dict"] = json.load(cache_file)
                return

        self.logger.debug(f"no cached transcription file found, running whisper transcribe with model: {self.transcription_model}")
        audio = whisper.load_audio(self.audio_filepath)
        model = whisper.load_model(self.transcription_model, device="cpu")
        result = whisper.transcribe(model, audio, language="en")

        self.logger.debug(f"transcription complete, performing post-processing cleanup")

        # Remove segments with no words, only music
        result["segments"] = [segment for segment in result["segments"] if segment["text"].strip() != "Music"]

        self.logger.debug(f"writing transcription data JSON to cache file: {whisper_cache_filepath}")
        with open(whisper_cache_filepath, "w") as cache_file:
            json.dump(result, cache_file, indent=4)

        self.outputs["transcription_data_dict"] = result

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
