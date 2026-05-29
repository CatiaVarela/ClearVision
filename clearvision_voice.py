"""Estimation de distance (m) et annonce vocale pour ClearVision."""

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pyttsx3

MAX_DISTANCE_M = 4.0
COOLDOWN_S = 2.5
REPEAT_BLOCK_S = 8.0
STABLE_FRAMES_REQUIRED = 2

# Calibration : distances calibrées pour une hauteur de référence (évite le biais 1080p).
REFERENCE_FRAME_HEIGHT = 720.0
FOCAL_LENGTH_FACTOR = 0.92

REAL_HEIGHT_M = {
    "person": 1.70,
    "dog": 0.50,
    "cat": 0.30,
    "car": 1.50,
    "bicycle": 1.10,
    "motorcycle": 1.20,
    "bus": 3.00,
    "truck": 3.00,
    "horse": 1.60,
    "cow": 1.40,
    "sheep": 0.90,
    "elephant": 3.00,
    "bear": 1.50,
    "bird": 0.30,
    "bench": 0.90,
    "chair": 0.90,
    "potted plant": 0.60,
    "stop sign": 2.10,
    "fire hydrant": 1.00,
    "backpack": 0.50,
}
DEFAULT_HEIGHT_M = 1.0

# Obstacles annonçables (pas la végétation HSV — distances non fiables).
VOICE_LABELS = {
    "person",
    "dog",
    "cat",
    "car",
    "bicycle",
    "motorcycle",
    "bus",
    "truck",
    "horse",
    "cow",
    "sheep",
    "elephant",
    "bear",
    "bird",
    "bench",
    "chair",
    "potted plant",
    "stop sign",
    "fire hydrant",
    "backpack",
    "handbag",
    "suitcase",
    "skateboard",
    "scooter",
}

# Jamais annoncer (boîtes « haie d'arbres » = fond de scène).
VOICE_EXCLUDED = {
    "tree_line",
    "hsv",
    "foliage",
    "hedge",
    "green bush",
    "shrub",
    "tree in park",
    "evergreen tree",
}

COCO_LABEL_FR = {
    "person": "personne",
    "bicycle": "vélo",
    "car": "voiture",
    "motorcycle": "moto",
    "bus": "bus",
    "truck": "camion",
    "dog": "chien",
    "cat": "chat",
    "horse": "cheval",
    "cow": "vache",
    "sheep": "mouton",
    "elephant": "éléphant",
    "bear": "ours",
    "bird": "oiseau",
    "bench": "banc",
    "chair": "chaise",
    "potted plant": "plante en pot",
    "stop sign": "panneau stop",
    "fire hydrant": "borne incendie",
    "backpack": "sac à dos",
    "handbag": "sac",
    "suitcase": "valise",
    "skateboard": "skateboard",
}


def label_to_fr(label_key: str, label_fr_map: dict | None = None) -> str:
    if label_fr_map and label_key in label_fr_map:
        return label_fr_map[label_key]
    return COCO_LABEL_FR.get(label_key, label_key.replace("_", " "))


def is_voice_eligible(label_key: str) -> bool:
    if label_key in VOICE_EXCLUDED:
        return False
    return label_key in VOICE_LABELS


def estimate_distance_m(bbox, frame_height: int, label_key: str) -> float:
    _x1, y1, _x2, y2 = bbox
    h_px = max(1, y2 - y1)
    real_h = REAL_HEIGHT_M.get(label_key, DEFAULT_HEIGHT_M)
    focal_px = REFERENCE_FRAME_HEIGHT * FOCAL_LENGTH_FACTOR
    return real_h * focal_px / h_px


def horizontal_sector(x_center: float, frame_width: int) -> str:
    r = x_center / frame_width
    if r < 0.35:
        return "à gauche"
    if r > 0.65:
        return "à droite"
    return "devant vous"


def format_distance_m(distance_m: float) -> str:
    d = max(0.5, round(distance_m))
    if d <= 1:
        return "à environ 1 mètre"
    return f"à environ {int(d)} mètres"


def _pick_french_voice(engine: pyttsx3.Engine) -> None:
    for voice in engine.getProperty("voices") or []:
        vid = (voice.id or "").lower()
        name = (voice.name or "").lower()
        if "fr" in vid or "fr" in name or "hortense" in name or "julie" in name:
            engine.setProperty("voice", voice.id)
            return


def synthesize_message_to_wav(message: str, wav_path: Path) -> None:
    """Génère un WAV (moteur neuf à chaque fois — évite le blocage pyttsx3 en boucle)."""
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", 165)
        _pick_french_voice(engine)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        engine.save_to_file(message, str(wav_path))
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass


class SpeechRecorder:
    """File d'annonces pour la vidéo : texte pendant l'analyse, WAV à la fin."""

    def __init__(self, work_dir: Path | None = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="clearvision_voice_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.pending: list[tuple[float, str]] = []
        self.clips: list[tuple[float, Path]] = []

    def queue(self, start_sec: float, message: str) -> None:
        self.pending.append((start_sec, message))

    def synthesize_all(self) -> None:
        self.clips.clear()
        for index, (start_sec, message) in enumerate(self.pending):
            wav_path = self.work_dir / f"clip_{index:04d}.wav"
            synthesize_message_to_wav(message, wav_path)
            if wav_path.exists() and wav_path.stat().st_size > 100:
                self.clips.append((start_sec, wav_path))

    @property
    def count(self) -> int:
        return len(self.clips)

    @property
    def pending_count(self) -> int:
        return len(self.pending)


def build_narration_wav(clips: list[tuple[float, Path]], duration_sec: float, out_wav: Path) -> None:
    """Assemble les phrases sur une piste silencieuse (nécessite pydub)."""
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise ImportError(
            "Installez pydub pour enregistrer la voix dans la vidéo : pip install pydub"
        ) from exc

    duration_ms = int(duration_sec * 1000) + 1500
    track = AudioSegment.silent(duration=duration_ms)

    for start_sec, wav_path in clips:
        if not wav_path.exists():
            continue
        segment = AudioSegment.from_wav(str(wav_path))
        track = track.overlay(segment, position=int(start_sec * 1000))

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    track.export(str(out_wav), format="wav")


def _find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg est requis pour intégrer la voix dans la vidéo. "
            "Installez : pip install imageio-ffmpeg"
        ) from exc


def mux_audio_into_video(video_path: Path, audio_wav: Path, output_path: Path) -> None:
    """Ajoute la piste audio à la vidéo."""
    ffmpeg = _find_ffmpeg()

    tmp_out = output_path.with_suffix(".mux.tmp.mp4")
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_wav),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(tmp_out),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg a échoué :\n{result.stderr}")

    tmp_out.replace(output_path)


class VoiceAnnouncer:
    """Synthèse vocale : obstacles proches, voix FR, anti-répétition."""

    def __init__(
        self,
        max_distance_m: float = MAX_DISTANCE_M,
        cooldown_s: float = COOLDOWN_S,
        playback: bool = True,
    ):
        self.max_distance_m = max_distance_m
        self.cooldown_s = cooldown_s
        self.playback = playback
        self._last_spoke = 0.0
        self._last_key = ""
        self._last_key_time = 0.0
        self._stable_key = ""
        self._stable_count = 0

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)
        _pick_french_voice(self.engine)

    def _speak(self, message: str, wav_path: Path | None = None) -> None:
        if wav_path is not None:
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            self.engine.save_to_file(message, str(wav_path))
            self.engine.runAndWait()
            if self.playback and wav_path.exists():
                self.engine.say(message)
                self.engine.runAndWait()
        elif self.playback:
            self.engine.say(message)
            self.engine.runAndWait()

    def _eligible(self, detections: list[dict], frame_height: int) -> list[dict]:
        out = []
        for det in detections:
            key = det["label_key"]
            if not is_voice_eligible(key):
                continue
            dist = det.get("distance_m")
            if dist is None:
                dist = estimate_distance_m(det["bbox"], frame_height, key)
            if dist <= self.max_distance_m:
                out.append({**det, "distance_m": dist})
        return out

    def _pick_closest(self, eligible: list[dict]) -> dict | None:
        if not eligible:
            return None
        return min(eligible, key=lambda d: d["distance_m"])

    def maybe_announce(
        self,
        detections: list[dict],
        frame_width: int,
        frame_height: int,
        *,
        record_at_sec: float | None = None,
        recorder: SpeechRecorder | None = None,
    ) -> str | None:
        if time.time() - self._last_spoke < self.cooldown_s:
            return None

        target = self._pick_closest(self._eligible(detections, frame_height))
        if target is None:
            self._stable_key = ""
            self._stable_count = 0
            return None

        x1, _y1, x2, _y2 = target["bbox"]
        x_center = (x1 + x2) / 2.0
        sector = horizontal_sector(x_center, frame_width)
        announce_key = f"{target['label_key']}|{sector}"

        if announce_key == self._stable_key:
            self._stable_count += 1
        else:
            self._stable_key = announce_key
            self._stable_count = 1

        if self._stable_count < STABLE_FRAMES_REQUIRED:
            return None

        now = time.time()
        if announce_key == self._last_key and now - self._last_key_time < REPEAT_BLOCK_S:
            return None

        label_fr = target["label_fr"]
        dist_phrase = format_distance_m(target["distance_m"])
        message = f"{label_fr.capitalize()}, {dist_phrase}, {sector}."

        if recorder is not None and record_at_sec is not None:
            recorder.queue(record_at_sec, message)
        elif self.playback:
            self._speak(message)

        self._last_spoke = now
        self._last_key = announce_key
        self._last_key_time = now
        return message
