"""Lab 3: paired LLM/VLM confusion detection and confusion-aware storytelling.

This is a *checkpoint-based* experimental prototype, not continuous emotion tracking.
It does not train a model and does not diagnose internal psychological states.

Quick test (no robot, no API charge):
  python confusion_storytelling.py --demo --participant TEST --condition assistance

Real robot + microphone:
  python confusion_storytelling.py --participant P01 --condition assistance \
    --story A --input mic --robot-ip 192.168.1.10
"""
from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any

LABELS = {"CONFUSED", "NOT_CONFUSED", "UNCERTAIN"}

STORIES = {
    "A": [
        {
            "narration": "Mira discovered a silver key beneath the lighthouse stairs. "
                         "She handed the key to Leo, who put it inside his blue map case. "
                         "They decided to search for the door it might open.",
            "prompt": "What do you think they should explore next?",
            "clarification": "Mira found the key, but Leo is now carrying it in his blue map case.",
        },
        {
            "narration": "They found a locked room next to the old signal tower. "
                         "Mira reached into her own bag, but could not find the key. "
                         "Leo was standing beside her holding the map case.",
            "prompt": "What should they try to do now?",
            "clarification": "Mira's bag is empty because she gave the key to Leo earlier. The key is in Leo's map case.",
        },
        {
            "narration": "Leo opened his map case. A folded note was lying on top of the key. "
                         "The note said the room could only be opened after the lighthouse light flashed twice.",
            "prompt": "How would you continue this story?",
            "clarification": "They found both the key and a note. The note says they must wait for two light flashes before opening the room.",
        },
    ],
    "B": [
        {
            "narration": "Ava discovered a handwritten letter at the town library. "
                         "She asked Jun to keep it safe, and Jun placed it in the red folder. "
                         "Together they started searching for its sender.",
            "prompt": "What could their first clue be?",
            "clarification": "Ava found the letter, but Jun is holding it now in the red folder.",
        },
        {
            "narration": "At the train station, Ava checked her backpack for the letter. "
                         "It was not there. Jun had brought the red folder and was waiting near the ticket machines.",
            "prompt": "What do you think they should do next?",
            "clarification": "The letter is not in Ava's backpack because Jun kept it in the red folder.",
        },
        {
            "narration": "Inside the red folder they found a sketch attached to the letter. "
                         "It showed a clock tower at sunset, with a star marking its north entrance.",
            "prompt": "How would you finish their next adventure?",
            "clarification": "The letter has an attached map. The star marks the north entrance of the clock tower at sunset.",
        },
    ],
}

CLASSIFICATION_PROMPT = """You analyze *observable signs* that a user is confused during collaborative storytelling.
Confusion here means difficulty understanding the narrative, participants, or story sequence.
Do NOT claim to know the person's internal mental state. Expressions alone are weak evidence.
Return ONLY a valid JSON object with exactly one field: {"label":"CONFUSED"} or
{"label":"NOT_CONFUSED"} or {"label":"UNCERTAIN"}.
CONFUSED: clear evidence of misunderstanding, a request for clarification, or expressed uncertainty about the story.
NOT_CONFUSED: clear evidence the user understands and continues without confusion indicators.
UNCERTAIN: insufficient, ambiguous, or missing evidence. Do not classify neutral expression as proof.
"""


@dataclass
class Prediction:
    label: str = "UNCERTAIN"
    milliseconds: float | None = None
    error: str = ""


def parse_label(raw: str) -> str:
    """Strictly parse a JSON label; never accidentally treat NOT_CONFUSED as CONFUSED."""
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        match = re.search(r"\{[^{}]*\}", raw or "")
        try:
            payload = json.loads(match.group(0)) if match else {}
        except ValueError:
            payload = {}
    label = payload.get("label", "") if isinstance(payload, dict) else ""
    return label if label in LABELS else "UNCERTAIN"


def jpg_as_data_url(frame: Any) -> str:
    import cv2
    height, width = frame.shape[:2]
    if width > 640:
        frame = cv2.resize(frame, (640, int(height * 640 / width)))
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    if not ok:
        raise RuntimeError("Could not encode camera frame as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def classify(client: Any, model: str, transcript: str, story_context: str, frame: Any = None) -> Prediction:
    started = time.perf_counter()
    try:
        prompt = f"Robot's most recent story segment:\n{story_context}\n\nUser's utterance:\n{transcript}\n\nClassify current evidence."
        content: Any = prompt
        if frame is not None:
            content = [
                {"type": "text", "text": prompt + "\nAlso consider visible cues, but avoid overinterpreting appearance."},
                {"type": "image_url", "image_url": {"url": jpg_as_data_url(frame)}},
            ]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        label = parse_label(response.choices[0].message.content or "")
        return Prediction(label, (time.perf_counter() - started) * 1000)
    except Exception as exc:
        return Prediction("UNCERTAIN", (time.perf_counter() - started) * 1000, str(exc)[:200])


def capture_robot_frame(mini: Any, seconds: float = 8.0) -> Any:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        frame = mini.media.get_frame()
        if frame is not None:
            return frame
        time.sleep(0.1)
    return None


def microphone_to_wav(mini: Any, seconds: float) -> io.BytesIO:
    """Record one timed utterance from the Reachy Mini mic, convert it to mono 16-bit WAV."""
    import numpy as np
    chunks = []
    mini.media.start_recording()
    try:
        print(f"  Recording {seconds:g} seconds. Speak now...")
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            sample = mini.media.get_audio_sample()
            if sample is not None:
                chunks.append(sample)
            else:
                time.sleep(0.01)
    finally:
        mini.media.stop_recording()
    if not chunks:
        raise RuntimeError("No microphone samples received. Check Reachy Mini media/daemon.")
    audio = np.concatenate(chunks, axis=0)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    rate = int(mini.media.get_input_audio_samplerate())
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    buf.name = "utterance.wav"
    return buf


def transcribe(client: Any, wav: io.BytesIO, model: str) -> str:
    result = client.audio.transcriptions.create(model=model, file=("utterance.wav", wav, "audio/wav"))
    return result.text.strip()


def say(client: Any, mini: Any, text: str, tts_model: str, voice: str, no_tts: bool, mock: bool) -> bool:
    """Speak via Reachy speaker; return whether robot speech was dispatched successfully."""
    print(f"\nReachy > {text}")
    if mock or no_tts or mini is None:
        return False  # printed or simulated, not real audible robot playback
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with client.audio.speech.with_streaming_response.create(
            model=tts_model, voice=voice, input=text, response_format="wav"
        ) as response:
            response.stream_to_file(out)
        mini.media.play_sound(out)  # Official teacher-demo SDK pattern.
        # Some playback backends return before the WAV has finished playing.
        # Keep the temporary file alive and avoid overlapping the next utterance.
        try:
            with wave.open(out, "rb") as wf:
                playback_seconds = wf.getnframes() / wf.getframerate()
            time.sleep(playback_seconds + 0.2)
        except (wave.Error, ZeroDivisionError):
            time.sleep(0.5)
        return True  # playback requested, not independently verified audible
    except Exception as exc:
        print(f"  [TTS/robot audio error] {exc}")
        return False
    finally:
        try:
            os.remove(out)
        except OSError:
            pass


def gentle_nod(mini: Any) -> None:
    if mini is None:
        return
    from reachy_mini.utils import create_head_pose
    try:
        mini.goto_target(head=create_head_pose(pitch=10, degrees=True), duration=0.3)
        mini.goto_target(head=create_head_pose(degrees=True), duration=0.3)
    except Exception as exc:
        print(f"  [Head motion unavailable] {exc}")


def explicit_request(text: str) -> bool:
    return bool(re.search(r"\b(explain|clarify|confused|don't understand|do not understand|what do you mean|lost)\b", text, re.I))


def mock_predict(transcript: str, visual: bool = False) -> Prediction:
    """DEMO ONLY. These are fake values for end-to-end software smoke testing."""
    if explicit_request(transcript) or re.search(r"\b(not sure|wait|why)\b", transcript, re.I):
        label = "CONFUSED"
    else:
        label = "NOT_CONFUSED"
    return Prediction(label, 0.0, "MOCK prediction - NOT research data")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rating(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if not value:
            return ""  # skip if participant declines
        if value in {"1", "2", "3", "4", "5", "6", "7"}:
            return value
        print("  Please enter 1–7 or press Enter to skip.")


def run(args: argparse.Namespace) -> Path:
    if args.demo and args.input == "mic":
        raise SystemExit("--demo uses keyboard input only. Omit --input mic.")
    if args.input == "mic" and args.no_robot:
        raise SystemExit("Microphone mode requires a connected robot.")
    if args.input == "mic" and args.demo:
        raise SystemExit("Demo mode cannot record robot microphone.")
    if not args.demo and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing. Create a local .env file (see README) or use --demo.")
    if not args.no_robot and not args.demo and not args.robot_ip:
        raise SystemExit("Set REACHY_IP in .env or pass --robot-ip, or use --no-robot.")

    client = None
    if not args.demo:
        from openai import OpenAI
        client = OpenAI()

    mini_context = None
    if not args.demo and not args.no_robot:
        from reachy_mini import ReachyMini
        mini_context = ReachyMini(host=args.robot_ip, connection_mode="network", media_backend="default")

    # User-provided file in keyboard/no-robot mode for vision comparison.
    still_frame = None
    if args.image:
        import cv2
        still_frame = cv2.imread(args.image)
        if still_frame is None:
            raise SystemExit(f"Cannot read image: {args.image}")

    stories = STORIES[args.story]
    rows: list[dict] = []
    started_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.output_dir) / f"{args.participant}_{args.story}_{args.condition}_{started_utc}.csv"
    print(f"\n=== Story {args.story}; {args.condition.upper()} condition ===")
    print("IMPORTANT: this is a small checkpoint-based pilot, not continuous real-time confusion detection.")
    if args.demo:
        print("*** MOCK MODE: zero API calls; predictions and output are synthetic ***")
    elif args.no_robot:
        print("NO ROBOT MODE: robot output is printed, not physically executed.")

    def session(mini: Any) -> None:
        if mini is not None:
            mini.enable_motors()
            gentle_nod(mini)
        say(client, mini, "Let's create a story together. I'll start, and then it's your turn.",
            args.tts_model, args.voice, args.no_tts, args.demo)

        for idx, stage in enumerate(stories, 1):
            story_text = stage["narration"] + " " + stage["prompt"]
            say(client, mini, story_text, args.tts_model, args.voice, args.no_tts, args.demo)
            print(f"\n--- Checkpoint {idx}/{len(stories)} ---")
            if args.input == "typed" or args.demo:
                transcript = input("Participant (type your response; q to quit)> ").strip()
                if transcript.lower() == "q":
                    break
                utterance_end = time.perf_counter()
            else:
                input("Press Enter when participant is ready to speak (starts timed recording)...")
                wav = microphone_to_wav(mini, args.record_seconds)
                utterance_end = time.perf_counter()
                try:
                    transcript = transcribe(client, wav, args.stt_model)
                except Exception as exc:
                    print(f"Speech transcription error: {exc}")
                    transcript = ""
                print(f"Recognized speech: {transcript!r}")
            # For mic mode, starts immediately after fixed-duration recording ends;
            # therefore this includes STT latency, frame capture, and model inference.
            input_end = utterance_end
            stt_ms = (time.perf_counter() - utterance_end) * 1000 if args.input == "mic" else 0.0
            frame = still_frame
            if mini is not None:
                try:
                    frame = capture_robot_frame(mini)
                except Exception as exc:
                    print(f"Camera error: {exc}")
            if frame is None:
                print("[No camera image. VLM prediction will be logged as MISSING.] ")

            # Both detection paths receive precisely the same accumulated story context.
            context = " ".join(s["narration"] for s in stories[:idx])
            if args.demo:
                llm = mock_predict(transcript)
                vlm = mock_predict(transcript, True) if frame is not None else Prediction(error="NO_IMAGE")
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    llm_job = pool.submit(classify, client, args.model, transcript, context)
                    vlm_job = (pool.submit(classify, client, args.model, transcript,
                                           context, frame) if frame is not None else None)
                    llm = llm_job.result()
                    # Do not wait for the VLM to trigger the LLM assistance response.
                    wants_help = explicit_request(transcript)
                    proactive = args.condition == "assistance" and llm.label == "CONFUSED" and not wants_help
                    triggered = proactive or wants_help
                    if triggered:
                        response_text = "Here's a quick recap: " + stage["clarification"]
                    else:
                        response_text = "Thanks for your idea. Let's see what happens next."
                    dispatch_ms = (time.perf_counter() - input_end) * 1000
                    speech_dispatched = say(client, mini, response_text, args.tts_model, args.voice,
                                            args.no_tts, args.demo)
                    gentle_nod(mini)
                    vlm = vlm_job.result() if vlm_job is not None else Prediction(error="NO_IMAGE")
            if args.demo:
                wants_help = explicit_request(transcript)
                proactive = args.condition == "assistance" and llm.label == "CONFUSED" and not wants_help
                triggered = proactive or wants_help
                response_text = ("Here's a quick recap: " + stage["clarification"] if triggered
                                 else "Thanks for your idea. Let's see what happens next.")
                dispatch_ms = (time.perf_counter() - input_end) * 1000
                speech_dispatched = say(client, mini, response_text, args.tts_model,
                                        args.voice, args.no_tts, True)
            print(f"[LLM: {llm.label}; VLM: {vlm.label if frame is not None else 'MISSING'}]")
            rows.append({
                "participant_id": args.participant,
                "session_id": started_utc,
                "story": args.story,
                "condition": args.condition,
                "checkpoint": idx,
                "model": args.model if not args.demo else "MOCK",
                "input_mode": args.input,
                "stt_ms": round(stt_ms, 1) if args.input == "mic" else "",
                "llm_label": llm.label,
                "vlm_label": vlm.label if frame is not None else "MISSING",
                "llm_model_ms": round(llm.milliseconds or 0, 1),
                "vlm_model_ms": round(vlm.milliseconds or 0, 1) if frame is not None else "",
                "model_error_llm": llm.error,
                "model_error_vlm": vlm.error,
                "explicit_help_requested": int(wants_help),
                "proactive_assistance": int(proactive),
                "response_kind": "clarification" if triggered else "neutral",
                "response_decision_ms_from_input_end": round(dispatch_ms, 1),
                "robot_speech_dispatched": int(speech_dispatched),
                "self_report_confusion_1to7": "",
                "self_report_support_1to7": "",
                "self_report_enjoyment_1to7": "",
                "self_report_continue_1to7": "",
                "comprehension_correct_out_of_3": "",
            })
        # Retrospective ratings. Keep these outside the story to avoid changing behavior mid-story.
        if rows and not args.skip_survey:
            print("\n=== Post-session survey (participant responses; Enter skips) ===")
            for row in rows:
                row["self_report_confusion_1to7"] = rating(
                    f"At checkpoint {row['checkpoint']}, how confused were you? (1–7): ")
            support = rating("I felt supported by Reachy Mini. (1–7): ")
            enjoyment = rating("The storytelling was enjoyable. (1–7): ")
            cont = rating("I would like to continue storytelling. (1–7): ")
            print("Give 3 identical-difficulty comprehension questions outside this script.")
            correct = input("How many answers were correct? (0–3, Enter to skip): ").strip()
            if correct not in {"0", "1", "2", "3"}:
                correct = ""
            for row in rows:
                row["self_report_support_1to7"] = support
                row["self_report_enjoyment_1to7"] = enjoyment
                row["self_report_continue_1to7"] = cont
                row["comprehension_correct_out_of_3"] = correct

    try:
        if mini_context is not None:
            with mini_context as mini:
                session(mini)
        else:
            session(None)
    finally:
        write_csv(out, rows)
        print(f"\nSaved trial log: {out}")
        print("No raw audio, transcripts, or participant photos were written to disk.")
    return out


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True, help="P01, P02, etc. Avoid real names.")
    parser.add_argument("--condition", choices=["baseline", "assistance"], required=True)
    parser.add_argument("--story", choices=["A", "B"], default="A")
    parser.add_argument("--input", choices=["typed", "mic"], default="typed", help="Mic uses Reachy mic + API transcription")
    parser.add_argument("--robot-ip", default=os.environ.get("REACHY_IP", ""))
    parser.add_argument("--no-robot", action="store_true", help="Test with OpenAI but no robot")
    parser.add_argument("--image", default="", help="Path to a single user image for testing without a robot")
    parser.add_argument("--demo", action="store_true", help="No robot, no API; synthetic predictions for checking flow")
    parser.add_argument("--no-tts", action="store_true", help="Do not play speech on the robot (console only)")
    parser.add_argument("--record-seconds", type=float, default=7.0)
    parser.add_argument("--model", default=os.environ.get("CONFUSION_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--stt-model", default="gpt-4o-mini-transcribe")
    parser.add_argument("--tts-model", default="gpt-4o-mini-tts")
    parser.add_argument("--voice", default="alloy")
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--skip-survey", action="store_true")
    args = parser.parse_args()
    if args.record_seconds <= 0 or args.record_seconds > 30:
        parser.error("--record-seconds must be between 0 and 30")
    if args.demo:
        args.no_robot = True
    run(args)


if __name__ == "__main__":
    main()
