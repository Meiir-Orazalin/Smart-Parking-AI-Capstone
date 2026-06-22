import cv2

from smart_parking_pipeline import ParkingSlotPipeline


def main():
    video_path = "CarPark.mp4"
    pipeline = ParkingSlotPipeline()

    # 1) Calibrate on first 20 frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(20):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")

    pipeline.calibrate_from_frames(frames, save_path="slots_auto.json")
    pipeline.load_slots("slots_auto.json")

    # 2) Process full video
    cap = cv2.VideoCapture(video_path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pipeline.process_frame(frame)
        annotated = pipeline.annotate_frame(frame, result["slots"])
        cv2.imshow("Smart Parking", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
