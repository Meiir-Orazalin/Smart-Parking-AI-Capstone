import os

import cv2
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import VideoFileSource, StreamConfig, VideoMetadata


def main():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Set ROBOFLOW_API_KEY env var before running.")

    workflow_id = os.getenv("ROBOFLOW_WORKFLOW_ID", "S4Q4b6wFRHG6HfIg3obu")
    # Use workspace ID from the embed token by default (works for workflows API)
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "RuvZNaEFZJNCsDP9bBG7D40vJC42")
    video_path = os.getenv("VIDEO_PATH", "CarPark.mp4")

    api_url = os.getenv("ROBOFLOW_API_URL", "https://api.roboflow.com")
    client = InferenceHTTPClient.init(
        api_url=api_url,
        api_key=api_key,
    )

    source = VideoFileSource(video_path)

    config = StreamConfig(
        stream_output=["visualization"],
        data_output=["predictions"],
        requested_plan="webrtc-gpu-medium",
        requested_region="us",
    )

    session = client.webrtc.stream(
        source=source,
        workflow=workflow_id,
        workspace=workspace,
        image_input="image",
        config=config,
    )

    @session.on_frame
    def show_frame(frame, metadata: VideoMetadata):
        cv2.imshow("Workflow Output", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            session.close()

    @session.on_data()
    def on_data(data: dict, metadata: VideoMetadata):
        print(f"Frame {metadata.frame_id}: {data}")

    session.run()


if __name__ == "__main__":
    main()
