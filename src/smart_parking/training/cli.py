from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import generate_dataset_yaml, prepare_dataset
from .frames import extract_frames_from_video
from .runner import SegmentationTrainingConfig, train_segmentation_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart Parking training utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    frames_parser = subparsers.add_parser("extract-frames", help="Extract labeling frames from a video.")
    frames_parser.add_argument("--video", required=True, help="Input video path.")
    frames_parser.add_argument("--output", required=True, help="Directory to save frames.")
    frames_parser.add_argument("--every-n-frames", type=int, default=30, help="Save every Nth frame.")
    frames_parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of saved frames.")
    frames_parser.add_argument("--start-frame", type=int, default=0, help="Start frame index.")
    frames_parser.add_argument("--end-frame", type=int, default=None, help="Optional end frame index.")
    frames_parser.add_argument("--prefix", default=None, help="Filename prefix for extracted frames.")
    frames_parser.add_argument("--image-ext", default=".jpg", help="Image extension for saved frames.")

    prep_parser = subparsers.add_parser("prepare-dataset", help="Build a YOLO-seg dataset from images and slot polygons.")
    prep_parser.add_argument("--images", required=True, help="Image file or directory.")
    prep_parser.add_argument("--slots", required=True, help="Slot JSON file or directory of per-image JSON files.")
    prep_parser.add_argument("--output", required=True, help="Dataset output directory.")
    prep_parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of samples assigned to train.")
    prep_parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    prep_parser.add_argument("--view-id", default=None, help="View ID to select from a slot cache.")
    prep_parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking them.")
    prep_parser.add_argument("--class-name", default="slot", help="Class name to place in the dataset YAML.")

    train_parser = subparsers.add_parser("train", help="Train a YOLO segmentation model.")
    train_parser.add_argument("--data-yaml", default=None, help="Dataset YAML path.")
    train_parser.add_argument("--dataset-root", default=None, help="Dataset root to auto-generate a YAML from.")
    train_parser.add_argument("--model", default="yolov8n-seg.pt", help="Base model to fine-tune.")
    train_parser.add_argument("--project", default="runs/train", help="Ultralytics project directory.")
    train_parser.add_argument("--name", default="parking_slot_segmentation", help="Run name.")
    train_parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    train_parser.add_argument("--device", default=None, help="Device string or index.")
    train_parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    train_parser.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    train_parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    train_parser.add_argument("--resume", action="store_true", help="Resume the previous run.")
    train_parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights.")
    train_parser.add_argument("--cache", action="store_true", help="Enable Ultralytics label caching.")
    train_parser.add_argument("--no-plots", action="store_true", help="Disable training plots.")
    train_parser.add_argument("--quiet", action="store_true", help="Reduce trainer output.")
    train_parser.add_argument("--optimizer", default=None, help="Optimizer name.")
    train_parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "extract-frames":
        result = extract_frames_from_video(
            args.video,
            args.output,
            every_n_frames=args.every_n_frames,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            prefix=args.prefix,
            image_ext=args.image_ext,
        )
        print(f"saved={result.total_saved} read={result.total_read} output={result.output_dir}")
        return 0

    if args.command == "prepare-dataset":
        prepared = prepare_dataset(
            args.images,
            args.slots,
            args.output,
            train_ratio=args.train_ratio,
            seed=args.seed,
            view_id=args.view_id,
            class_names=(args.class_name,),
            copy_images=args.copy_images,
        )
        print(
            f"dataset={prepared.root} yaml={prepared.dataset_yaml} train={prepared.train_count} val={prepared.val_count}"
        )
        return 0

    if args.command == "train":
        config = SegmentationTrainingConfig(
            model=args.model,
            data_yaml=args.data_yaml,
            dataset_root=args.dataset_root,
            project=args.project,
            name=args.name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            seed=args.seed,
            pretrained=not args.no_pretrained,
            cache=args.cache,
            resume=args.resume,
            plots=not args.no_plots,
            verbose=not args.quiet,
            optimizer=args.optimizer,
            lr0=args.lr0,
        )
        results = train_segmentation_model(config)
        print(results)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

