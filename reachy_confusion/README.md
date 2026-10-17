# Lab 3 — Reachy Mini Confusion-Aware Storytelling

A **checkpoint-based pilot prototype** for Cornell Tech INFO 5356 Lab 3. This is **not** a trained emotion classifier and it does not run continuous face tracking. It uses a pretrained multimodal OpenAI model twice on the **same** story checkpoint:

- **LLM prediction** = story context + user's spoken/transcribed words
- **VLM prediction** = **the exact same** story context + words **plus one Reachy Mini camera frame**

Only the LLM prediction can trigger proactive clarification in the assistance condition. Both model labels are recorded for the language-only vs. multimodal comparison. The baseline condition tells the same story but does not provide *proactive* help; direct user requests for clarification are answered in both conditions. No model training is needed.

The supplied stories are **starter pilot materials**. Research partners should review whether the content and comprehension questions match the protocol before gathering official data. Don't mix demo/mock rows into research analyses.

## Files

- `confusion_storytelling.py`: stand-alone study runner; does **not** change the group's `run_file.tex` or the professor's demo files.
- `.env.example`: local API and robot settings (make an untracked `.env`).
- `requirements.txt`: Python dependencies, independent of the repository root's shell script named `requirements.txt`.
- `.gitignore`: protects `.env`, temporary media and participant log files.
- `test_confusion_storytelling.py`: simple logic tests (no hardware or API needed).

## 1. Windows PowerShell setup (Python 3.12)

Open PowerShell **inside `HRI-lab333/reachy_confusion`**, not `C:\\WINDOWS\\System32`:

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Edit the `.env` file to include your **real** `OPENAI_API_KEY` and the Reachy Mini address. Do not commit or screenshot your key. API calls and images sent for analysis may incur charges and involve uploading participant data to the API provider.

### Check the logic first (no robot, no API charge)

```powershell
.\\.venv\\Scripts\\python.exe -m unittest test_confusion_storytelling.py
.\\.venv\\Scripts\\python.exe confusion_storytelling.py --demo --participant TEST --condition assistance --story A --skip-survey
```

The `--demo` option produces **fake rule-based predictions** and never accesses the camera or API. It's only for testing the workflow, not for evaluating the model. Type a participant reply after each story segment.

### Check OpenAI with no robot (text-only LLM)

```powershell
.\\.venv\\Scripts\\python.exe confusion_storytelling.py --no-robot --participant PILOT --condition assistance --story A --input typed
```

To also test VLM, add `--image C:\\path\\to\\consented_photo.jpg`; otherwise the VLM value is marked `MISSING` (not a negative prediction). You can replace the photo with an actual Reachy Mini camera frame once connected.

### Run with Reachy Mini + typed input (hardware/speaker/camera check)

Connect your computer and robot to the same reachable network. Check the daemon before running:

```powershell
curl.exe http://192.168.1.10:8000/api/daemon/status
```

Replace the sample IP with your robot IP. If the daemon is on the laptop itself (Reachy Mini Lite), the address may be `localhost` and the daemon must be running.

```powershell
.\\.venv\\Scripts\\python.exe confusion_storytelling.py --participant P01 --condition baseline --story A --input typed --robot-ip 192.168.1.10
```

The robot narrates through its speaker (OpenAI TTS), the participant *types* a response, both models analyze the checkpoint, and the robot replies. In the baseline, confusion detection still runs for logging but it cannot proactively trigger clarification.

### Run with Reachy Mini microphone + camera (real interaction)

```powershell
.\\.venv\\Scripts\\python.exe confusion_storytelling.py --participant P01 --condition assistance --story B --input mic --robot-ip 192.168.1.10 --record-seconds 7
```

The operator presses Enter when the participant is ready; the microphone records for a fixed **7 seconds** (adjust 5–12 after pilot testing). Speech-to-text uses `gpt-4o-mini-transcribe`. This is a **push-to-record trial**, not voice activity detection. The photo is captured just after the utterance; check that Reachy's camera actually sees the participant's face before the formal study.

You can pass `--no-tts` if you only want printed speech while debugging. The robot will not provide audible narration in that mode, so **do not** mix those sessions with your formal experimental data.

## 2. Suggested participant assignment

Each participant completes **both** conditions, using story A for one and story B for the other. Counterbalance condition and story order among participants, e.g. P01 baseline/A then assistance/B, P02 assistance/A then baseline/B. Include both the verbal and visual prediction at each of three checkpoints. Allow a pause and the participant's free storytelling contribution after each robot prompt.

Use the same response policy in every condition. In the baseline, a participant who **explicitly** asks for an explanation receives the same prepared factual recap. In the assistance condition, an LLM `CONFUSED` label can trigger that recap even if they did not explicitly ask. A prediction of `UNCERTAIN` does *not* automatically trigger assistance.

## 3. What is (and is not) logged

Each run produces `logs/P01_A_baseline_YYYYMMDDTHHMMSSZ.csv` with labels, model timing, response type, logged software time to response decision, and post-session participant confusion and experience ratings. No raw transcript, images or audio are saved by default. The *estimated* `response_decision_ms_from_input_end` begins after typed input is submitted or the **fixed microphone recording ends**, includes any speech transcription, frame acquisition and LLM inference, and ends when the response is selected. It **does not include** speech-synthesis/network playback or complete audible-response time. Fixed recording time is excluded. The separate `stt_ms` is also provided for microphone mode. Do **not** call this an end-to-end audible-response latency measure in the report without instrumenting it further.

Collect three actual story comprehension answers with your research team and score them consistently, blinded to the condition if possible. Self-reported confusion is retrospective and may not be fully accurate: acknowledge this as a limitation. Protect logs containing participant IDs according to your agreed team procedure.

Do **not** claim facial expressions prove confusion; VLM predictions concern visible evidence of possible confusion.

## 4. Research status and limitations

- **Model:** same default API model (`gpt-4.1-mini`) in both conditions; confirm your account has access. Override with `--model MODEL` or `CONFUSION_MODEL` in `.env`.
- **Audio:** uses official `mini.media.start_recording()`, `get_audio_sample()`, and `get_input_audio_samplerate()`. SDK media backends vary across Reachy Mini versions. We cannot remotely validate your hardware, firewall, microphone, or speaker.
- **Robot:** uses `ReachyMini(host=..., connection_mode="network", media_backend="default")`, `enable_motors()`, `get_frame()`, `goto_target()`, `play_sound()` as in the professor's demo. Move cautiously, clear nearby objects, stop if motors behave unexpectedly.
- **Response:** simple scripted narrative and factual clarification at three checkpoints. It is not unrestricted LLM story generation and is not an always-on conversational agent.
- **Visual:** static images, captured after the user completes an utterance; one frame misses dynamic hesitation or fleeting expressions.
- **Training:** none. No dedicated VLA model is used.
- **Data privacy:** obtain explicit informed consent before recording or sending speech/photos to OpenAI. API handles media temporarily for inference; local code defaults to no raw media storage.
- **Comprehension:** write and pilot matching questions per story; the script prompts you for a single correct-answer count 0–3, it does not automatically grade answers.

## 5. Troubleshooting

- `ModuleNotFoundError`: install dependencies in the same `.venv` as the Python executable.
- `OPENAI_API_KEY missing`: verify `.env` is beside this README and you're running from this directory.
- Connection timeout: confirm IP, port 8000, daemon running, same reachable network; see the course `SETUP.md`.
- No camera frame: check that the robot's view faces the participant, verify media backend; VLM logs `MISSING` if unavailable.
- No microphone samples: test the Reachy Mini SDK's official `sound_record.py` example separately; some Windows GStreamer setups may require extra configuration. You can run the pilot in typed mode meanwhile.
- TTS may take longer than detection and the speaker pipeline may need separate troubleshooting. `robot_speech_dispatched=1` means the audio call returned without error, **not** proof the participant heard it.

## 6. Sources / starting point

- Class starter: https://github.com/Cornell-Tech-HRI-Students-2026/lab3 (`common.py`, `demo1_llm.py`, `demo2_vlm.py`)
- Reachy SDK audio recording: https://huggingface.co/docs/reachy_mini/examples/sound_record
- Reachy SDK media: https://huggingface.co/docs/reachy_mini/SDK/python-sdk

**Before real participant testing:** check informed consent, capture visibility, both API predictions, audible voice, and non-interruption of robot speech. Then lock prompts/model/settings for all official trials.
