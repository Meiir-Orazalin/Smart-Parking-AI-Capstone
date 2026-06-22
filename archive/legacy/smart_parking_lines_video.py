"""
Helper script: run the existing LineParkingDetector on frames from a video.

This does NOT introduce a new detection method – it simply:
1. Opens a video file (e.g. CarPark.mp4)
2. Samples every Nth frame
3. Saves each sampled frame as an image
4. Calls LineParkingDetector.process_image(...) on that image

Usage:
    python smart_parking_lines_video.py CarPark.mp4
    python smart_parking_lines_video.py CarPark.mp4 --stride 60 --max_frames 10
"""

import argparse
import os
from pathlib import Path

import cv2

from smart_parking_lines import LineParkingDetector


def main():
    parser = argparse.ArgumentParser(
        description="Run LineParkingDetector on sampled frames from a video file."
    )
    parser.add_argument("video_path", help="Path to the video file (e.g. CarPark.mp4)")
    parser.add_argument(
        "--stride",
        type=int,
        default=60,
        help="Process every Nth frame (default: 60).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=5,
        help="Maximum number of frames to process (default: 5).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live OpenCV window with annotated frames.",
    )

    args = parser.parse_args()

    video_path = args.video_path
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    print("============================================================")
    print("  🅿️  SMART PARKING AI - LINE DETECTION (VIDEO SAMPLER)")
    print("  Using existing LineParkingDetector on video frames")
    print("============================================================")

    detector = LineParkingDetector()

    frame_idx = 0
    processed = 0
    base = Path(video_path).stem

    while processed < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        frame_filename = f"{base}_frame_{frame_idx:05d}.jpg"
        frame_path = Path(frame_filename)
        cv2.imwrite(str(frame_path), frame)

        output_filename = f"{base}_detection_{frame_idx:05d}.jpg"
        print(f"\n🎞  Processing frame {frame_idx} -> {output_filename}")

        # Reuse the existing image-based pipeline.
        output, _ = detector.process_image(
            str(frame_path),
            output_path=str(output_filename),
            show=False,
            debug=False,
        )

        if args.display:
            # Show the annotated frame as a live video stream.
            cv2.imshow("Smart Parking AI - Line Detection (Video)", output)
            # Small delay; exit if user presses 'q'.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        processed += 1
        frame_idx += 1

    cap.release()

    if processed == 0:
        print("⚠️  No frames were processed. Try reducing --stride.")
    else:
        print(
            f"\n✅ Done. Generated {processed} annotated frame(s) from {video_path} "
            f"(look for *{base}_detection_*.jpg)."
        )


if __name__ == "__main__":
    main()

