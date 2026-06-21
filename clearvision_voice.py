"""Estimation de distance (m) et annonce vocale pour ClearVision."""

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pyttsx3

MAX_DISTANCE_M = 4.0
MAX_OBSTACLES_DISPLAY = 8
COOLDOWN_S = 2.5
REPEAT_BLOCK_S = 8.0
STABLE_FRAMES_REQUIRED = 2
MAX_VIDEO_ANNOUNCEMENTS = 30

# Calibration distance (modèle pinhole simplifié)
REFERENCE_FRAME_HEIGHT = 720.0
FOCAL_LENGTH_FACTOR = 0.92
# Webcam portable : champ plus large, sujet souvent en gros plan (visage/buste)
WEBCAM_FOCAL_LENGTH_FACTOR = 0.38

REAL_HEIGHT_M = {
    "person": 1.70,
    "person_group": 1.70,
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
    "deer": 1.20,
    "squirrel": 0.25,
    "rabbit": 0.35,
    "fox": 0.45,
    "duck": 0.40,
    "lion": 1.20,
    "tiger": 1.10,
    "tree_line": 3.00,
    "hedge": 1.50,
    "foliage": 2.00,
    "tree in park": 4.00,
    "flower": 0.40,
}
DEFAULT_HEIGHT_M = 1.0

# Obstacles annonçables (pas la végétation HSV — distances non fiables).
VOICE_LABELS = {
    "person",
    "person_group",
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
    "person_group": "groupe de personnes",
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
    "deer": "cerf",
    "squirrel": "écureuil",
    "rabbit": "lapin",
    "fox": "renard",
    "fish": "poisson",
    "butterfly": "papillon",
    "duck": "canard",
    "lion": "lion",
    "tiger": "tigre",
    "monkey": "singe",
    "snake": "serpent",
    "frog": "grenouille",
    "tree_line": "arbre",
    "hedge": "haie",
    "foliage": "végétation",
    "tree in park": "arbre",
    "evergreen tree": "arbre",
    "flower": "fleur",
    "grass": "herbe",
}

try:
    from config import LABEL_FR as CONFIG_LABEL_FR

    COCO_LABEL_FR = {**CONFIG_LABEL_FR, **COCO_LABEL_FR}
except ImportError:
    pass


def label_to_fr(label_key: str, label_fr_map: dict | None = None) -> str:
    if label_fr_map and label_key in label_fr_map:
        return label_fr_map[label_key]
    return COCO_LABEL_FR.get(label_key, label_key.replace("_", " "))


def is_voice_eligible(label_key: str) -> bool:
    if label_key in VOICE_EXCLUDED:
        return False
    if label_key in VOICE_LABELS:
        return True
    if label_key in REAL_HEIGHT_M:
        return True
    return label_key in COCO_LABEL_FR


def is_path_obstacle(label_key: str, source: str = "coco_or_world") -> bool:
    """
    Obstacle susceptible de gêner une personne (à afficher / annoncer).
    Exclut végétation de fond (arbres sur murs, haies HSV, plantes).
    """
    if source == "vegetation":
        return False
    if label_key in ("person", "person_group"):
        return True

    try:
        from object_learning.learned_labels import learned_object_names

        if label_key.lower() in learned_object_names():
            return True
    except ImportError:
        pass

    return is_voice_eligible(label_key)


def _effective_real_height_m(label_key: str, bbox: tuple, frame_height: int) -> float:
    """
    Hauteur réelle estimée de l'objet.
    Pour une personne en gros plan (webcam), la boîte ne couvre que le buste/visage :
    utiliser une hauteur effective plus petite évite de surestimer la distance.
    """
    real_height_m = REAL_HEIGHT_M.get(label_key, DEFAULT_HEIGHT_M)
    if label_key == "person_group":
        return real_height_m
    if label_key != "person":
        return real_height_m

    _x1, y1, _x2, y2 = bbox
    bounding_box_height_ratio = (y2 - y1) / max(1, frame_height)

    if bounding_box_height_ratio > 0.55:
        return 0.40
    if bounding_box_height_ratio > 0.35:
        return 0.65
    if bounding_box_height_ratio > 0.22:
        return 1.05
    return real_height_m


def estimate_distance_m(
    bbox,
    frame_height: int,
    label_key: str,
    *,
    focal_length_factor: float | None = None,
) -> float:
    _x1, y1, _x2, y2 = bbox
    bounding_box_height_pixels = max(1, y2 - y1)
    real_height_m = _effective_real_height_m(label_key, bbox, frame_height)
    focal_factor = focal_length_factor if focal_length_factor is not None else FOCAL_LENGTH_FACTOR
    # Focale proportionnelle à la hauteur réelle de l'image (pas une valeur 720p fixe)
    focal_length_pixels = frame_height * focal_factor
    return real_height_m * focal_length_pixels / bounding_box_height_pixels


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
        if len(self.pending) >= MAX_VIDEO_ANNOUNCEMENTS:
            return
        self.pending.append((start_sec, message))

    def dedupe_pending(self) -> None:
        """Supprime les messages identiques (garde la première occurrence)."""
        seen_messages: set[str] = set()
        unique_pending: list[tuple[float, str]] = []
        for start_sec, message in self.pending:
            if message in seen_messages:
                continue
            seen_messages.add(message)
            unique_pending.append((start_sec, message))
        self.pending = unique_pending[:MAX_VIDEO_ANNOUNCEMENTS]

    def synthesize_all(self) -> None:
        self.clips.clear()
        total_count = len(self.pending)
        for index, (start_sec, message) in enumerate(self.pending):
            if total_count > 3:
                print(f"  Synthèse vocale {index + 1}/{total_count}...")
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
        self._last_spoke_video_sec = -999.0
        self._last_key = ""
        self._last_key_time = 0.0
        self._last_key_video_sec = -999.0
        self._stable_key = ""
        self._stable_count = 0

    def _speak(self, message: str) -> None:
        """
        Parle un message à voix haute.
        Moteur pyttsx3 neuf à chaque appel — évite le blocage après la 1re annonce.
        """
        if not self.playback:
            return

        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", 165)
            _pick_french_voice(engine)
            engine.say(message)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass

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
        using_video_timeline = recorder is not None and record_at_sec is not None

        if using_video_timeline:
            if record_at_sec - self._last_spoke_video_sec < self.cooldown_s:
                return None
        elif time.time() - self._last_spoke < self.cooldown_s:
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

        if using_video_timeline:
            if (
                announce_key == self._last_key
                and record_at_sec - self._last_key_video_sec < REPEAT_BLOCK_S
            ):
                return None
        else:
            now = time.time()
            if announce_key == self._last_key and now - self._last_key_time < REPEAT_BLOCK_S:
                return None

        label_fr = target["label_fr"]
        dist_phrase = format_distance_m(target["distance_m"])
        message = f"{label_fr.capitalize()}, {dist_phrase}, {sector}."

        if using_video_timeline:
            recorder.queue(record_at_sec, message)
            self._last_spoke_video_sec = record_at_sec
            self._last_key = announce_key
            self._last_key_video_sec = record_at_sec
        else:
            self._speak(message)
            now = time.time()
            self._last_spoke = now
            self._last_key = announce_key
            self._last_key_time = now

        return message
