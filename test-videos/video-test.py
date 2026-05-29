import argparse
import importlib.util
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO, YOLOWorld

# --- CONFIG ---
DEFAULT_VIDEO = Path("../videos/38805-418875307.mp4")
YOLO_COCO_PATH = "../yolo11n.pt"
YOLO_WORLD_PATH = "../test-images/yolov8s-world.pt"
PROGRESS_EVERY = 30

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))

from clearvision_voice import (
    MAX_DISTANCE_M,
    SpeechRecorder,
    VoiceAnnouncer,
    build_narration_wav,
    mux_audio_into_video,
)

_YOLO_WORLD_SCRIPT = _ROOT / "test-images" / "yolo-world-test.py"


def _load_yolo_world_module():
    spec = importlib.util.spec_from_file_location("yolo_world_test", _YOLO_WORLD_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {_YOLO_WORLD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["yolo_world_test"] = module
    spec.loader.exec_module(module)
    return module


def result_path_for(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_yoloworld{video_path.suffix}")


VOICE_FRAME_STRIDE = 2


def main(
    video_path: Path,
    show_gui: bool = True,
    fast: bool = False,
    voice: bool = False,
):
    yw = _load_yolo_world_module()
    process_frame = yw.process_frame

    if voice and not fast:
        print("Mode rapide activé automatiquement avec --voice.")
        fast = True

    recorder = SpeechRecorder() if voice else None
    announcer = (
        VoiceAnnouncer(max_distance_m=MAX_DISTANCE_M, playback=False) if voice else None
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Vidéo introuvable : {video_path.resolve()}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = result_path_for(video_path)
    video_tmp = (
        out_path.with_name(f"{out_path.stem}_silent{out_path.suffix}") if voice else out_path
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_tmp), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'écrire la vidéo : {out_path.resolve()}")

    yolo_model = YOLO(YOLO_COCO_PATH)
    world_model = None if fast else YOLOWorld(YOLO_WORLD_PATH)

    frame_idx = 0
    hsv_frames = 0
    veg_total = 0
    t0 = time.perf_counter()

    print(f"Traitement : {video_path.resolve()}")
    if fast:
        print("Mode rapide : YOLO-World désactivé (voir --fast)")
    if voice:
        print(f"Annonce vocale : objets à au plus {MAX_DISTANCE_M:.0f} m")
        print("La voix sera intégrée dans le fichier MP4 exporté.")
    if total_frames > 0:
        print(f"Frames : {total_frames} — {width}x{height} @ {fps:.1f} fps")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        annotated, info = process_frame(
            frame,
            yolo_model,
            world_model,
            fast=fast,
            estimate_distances=voice,
        )
        writer.write(annotated)

        if announcer is not None and frame_idx % VOICE_FRAME_STRIDE == 0:
            h, w = frame.shape[:2]
            message = announcer.maybe_announce(
                info["detections"],
                w,
                h,
                record_at_sec=frame_idx / fps,
                recorder=recorder,
            )
            if message:
                print(f"  [voix] {message}")

        veg_total += info["veg_drawn"]
        if info["used_hsv"]:
            hsv_frames += 1

        frame_idx += 1
        if frame_idx % PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - t0
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0
            if total_frames > 0:
                pct = 100.0 * frame_idx / total_frames
                print(f"  {frame_idx}/{total_frames} ({pct:.0f} %) — {fps_proc:.1f} img/s")
            else:
                print(f"  frame {frame_idx} — {fps_proc:.1f} img/s")

        if show_gui:
            try:
                preview = cv2.resize(annotated, (1000, 800))
                cv2.imshow("ClearVision - YOLO-World (vidéo)", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Arrêt anticipé (touche q).")
                    break
            except cv2.error:
                show_gui = False

    cap.release()
    writer.release()
    if show_gui:
        cv2.destroyAllWindows()

    duration_sec = frame_idx / fps if fps > 0 else 0.0

    if voice and recorder is not None:
        if recorder.pending_count > 0:
            print(f"Synthèse vocale de {recorder.pending_count} annonce(s)...")
            recorder.synthesize_all()
        if recorder.count > 0:
            try:
                narr_wav = recorder.work_dir / "narration.wav"
                build_narration_wav(recorder.clips, duration_sec, narr_wav)
                print("Fusion audio + vidéo...")
                mux_audio_into_video(video_tmp, narr_wav, out_path)
                video_tmp.unlink(missing_ok=True)
                print(f"Voix intégrée : {recorder.count} annonce(s) dans la vidéo.")
            except (ImportError, RuntimeError) as exc:
                print(f"Attention : voix non intégrée — {exc}")
                if video_tmp.exists() and not out_path.exists():
                    video_tmp.rename(out_path)
        elif video_tmp.exists():
            video_tmp.rename(out_path)

    elapsed = time.perf_counter() - t0
    print(f"Vidéo enregistrée : {out_path.resolve()}")
    print(f"Frames traitées : {frame_idx} en {elapsed:.1f} s")
    if frame_idx:
        print(f"Végétation (moy.) : {veg_total / frame_idx:.1f} zone(s)/frame")
        print(f"Secours HSV : {hsv_frames}/{frame_idx} frame(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Détection COCO + arbres/buissons sur une vidéo"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Chemin de la vidéo à analyser",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Sauvegarder uniquement, sans prévisualisation OpenCV",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Sans YOLO-World (≈5× plus rapide) ; végétation via haie d'arbres seulement",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help=f"Voix intégrée dans le MP4 (détections ≤ {MAX_DISTANCE_M:.0f} m)",
    )
    args = parser.parse_args()
    main(
        video_path=args.video,
        show_gui=not args.no_gui,
        fast=args.fast,
        voice=args.voice,
    )
