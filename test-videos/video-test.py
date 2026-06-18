import argparse
import sys
import time
from pathlib import Path

import cv2

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
from config import DEFAULT_VIDEO, YOLO_COCO_PATH, YOLO_WORLD_PATH
from detection import load_coco_model, load_world_model, process_frame

PROGRESS_EVERY = 30
VOICE_FRAME_STRIDE = 2


def result_path_for(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_yoloworld{video_path.suffix}")


def main(
    video_path: Path,
    show_gui: bool = True,
    fast: bool = False,
    voice: bool = False,
):
    if voice and not fast:
        print("Mode rapide activé automatiquement avec --voice.")
        fast = True

    speech_recorder = SpeechRecorder() if voice else None
    voice_announcer = (
        VoiceAnnouncer(max_distance_m=MAX_DISTANCE_M, playback=False) if voice else None
    )

    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Vidéo introuvable : {video_path.resolve()}")

    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = result_path_for(video_path)
    silent_video_path = (
        output_path.with_name(f"{output_path.stem}_silent{output_path.suffix}") if voice else output_path
    )
    video_writer = cv2.VideoWriter(
        str(silent_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frames_per_second,
        (frame_width, frame_height),
    )
    if not video_writer.isOpened():
        video_capture.release()
        raise RuntimeError(f"Impossible d'écrire la vidéo : {output_path.resolve()}")

    yolo_model = load_coco_model()
    world_model = None if fast else load_world_model()

    frame_index = 0
    hsv_frame_count = 0
    vegetation_total = 0
    processing_start = time.perf_counter()

    print(f"Traitement : {video_path.resolve()}")
    print(f"Modèles : {YOLO_COCO_PATH.name}, {YOLO_WORLD_PATH.name if not fast else 'HSV seul'}")
    if fast:
        print("Mode rapide : YOLO-World désactivé (voir --fast)")
    if voice:
        print(f"Annonce vocale : objets à au plus {MAX_DISTANCE_M:.0f} m")
    if total_frame_count > 0:
        print(f"Frames : {total_frame_count} — {frame_width}x{frame_height} @ {frames_per_second:.1f} fps")

    while video_capture.isOpened():
        frame_read_ok, frame = video_capture.read()
        if not frame_read_ok:
            break

        annotated_frame, detection_info = process_frame(
            frame,
            yolo_model,
            world_model,
            fast=fast,
            estimate_distances=voice,
        )
        video_writer.write(annotated_frame)

        if voice_announcer is not None and frame_index % VOICE_FRAME_STRIDE == 0:
            frame_height, frame_width = frame.shape[:2]
            spoken_message = voice_announcer.maybe_announce(
                detection_info["detections"],
                frame_width,
                frame_height,
                record_at_sec=frame_index / frames_per_second,
                recorder=speech_recorder,
            )
            if spoken_message:
                print(f"  [voix] {spoken_message}")

        vegetation_total += detection_info["veg_drawn"]
        if detection_info["used_hsv"]:
            hsv_frame_count += 1

        frame_index += 1
        if frame_index % PROGRESS_EVERY == 0:
            elapsed_seconds = time.perf_counter() - processing_start
            processing_fps = frame_index / elapsed_seconds if elapsed_seconds > 0 else 0.0
            if total_frame_count > 0:
                progress_percent = 100.0 * frame_index / total_frame_count
                print(f"  {frame_index}/{total_frame_count} ({progress_percent:.0f} %) — {processing_fps:.1f} img/s")
            else:
                print(f"  frame {frame_index} — {processing_fps:.1f} img/s")

        if show_gui:
            try:
                preview_frame = cv2.resize(annotated_frame, (1000, 800))
                cv2.imshow("ClearVision - Vidéo", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Arrêt anticipé (touche q).")
                    break
            except cv2.error:
                show_gui = False

    video_capture.release()
    video_writer.release()
    if show_gui:
        cv2.destroyAllWindows()

    video_duration_seconds = frame_index / frames_per_second if frames_per_second > 0 else 0.0

    if voice and speech_recorder is not None:
        if speech_recorder.pending_count > 0:
            print(f"Synthèse vocale de {speech_recorder.pending_count} annonce(s)...")
            speech_recorder.synthesize_all()
        if speech_recorder.count > 0:
            try:
                narration_wav_path = speech_recorder.work_dir / "narration.wav"
                build_narration_wav(speech_recorder.clips, video_duration_seconds, narration_wav_path)
                print("Fusion audio + vidéo...")
                mux_audio_into_video(silent_video_path, narration_wav_path, output_path)
                silent_video_path.unlink(missing_ok=True)
                print(f"Voix intégrée : {speech_recorder.count} annonce(s) dans la vidéo.")
            except (ImportError, RuntimeError) as error:
                print(f"Attention : voix non intégrée — {error}")
                if silent_video_path.exists() and not output_path.exists():
                    silent_video_path.rename(output_path)
        elif silent_video_path.exists():
            silent_video_path.rename(output_path)

    elapsed_seconds = time.perf_counter() - processing_start
    print(f"Vidéo enregistrée : {output_path.resolve()}")
    print(f"Frames traitées : {frame_index} en {elapsed_seconds:.1f} s")
    if frame_index:
        print(f"Végétation (moy.) : {vegetation_total / frame_index:.1f} zone(s)/frame")
        print(f"Secours HSV : {hsv_frame_count}/{frame_index} frame(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Détection universelle sur vidéo")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Vidéo à analyser")
    parser.add_argument("--no-gui", action="store_true", help="Sans prévisualisation OpenCV")
    parser.add_argument("--fast", action="store_true", help="Sans YOLO-World (plus rapide)")
    parser.add_argument(
        "--voice",
        action="store_true",
        help=f"Voix intégrée dans le MP4 (détections ≤ {MAX_DISTANCE_M:.0f} m)",
    )
    arguments = parser.parse_args()
    main(
        video_path=arguments.video,
        show_gui=not arguments.no_gui,
        fast=arguments.fast,
        voice=arguments.voice,
    )
