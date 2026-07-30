"""Fine-tune YOLO on VisDrone for top-down drone detection.

VisDrone2019 is the standard aerial benchmark. Its 10 classes are exactly what
a quadcopter sees outdoors:

    pedestrian, people, bicycle, car, van, truck, tricycle,
    awning-tricycle, bus, motor

Ultralytics ships a `VisDrone.yaml` that AUTO-DOWNLOADS the dataset (~2 GB) on
first run, so you don't have to fetch it manually.

    python train_visdrone.py --model yolo11s.pt --epochs 60 --imgsz 1024

WARNING: training is a GPU job. On this CPU-only box it would take days — run it
on a machine with an NVIDIA GPU (or a cloud instance), then copy the resulting
`best.pt` here and use it:

    python -m aerial -i footage.mp4 -o out.mp4 --weights models/visdrone.pt

The trained model plugs straight into the detector; no code changes needed.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description="Fine-tune YOLO on VisDrone")
    ap.add_argument("--model", default="yolo11s.pt",
                    help="base weights to fine-tune (n/s/m/l/x)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="larger helps tiny aerial objects (1024-1536 typical)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0",
                    help="GPU id(s), e.g. 0 or 0,1. 'cpu' is impractically slow")
    ap.add_argument("--data", default="VisDrone.yaml",
                    help="dataset yaml (bundled with ultralytics; auto-downloads)")
    ap.add_argument("--name", default="visdrone_yolo11s")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, name=args.name,
        patience=15, close_mosaic=10)

    # quick validation report
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print("mAP50-95:", metrics.box.map)
    print("mAP50   :", metrics.box.map50)
    print(f"\nBest weights: runs/detect/{args.name}/weights/best.pt")
    print("Copy it to models/visdrone.pt and run:")
    print("  python -m aerial -i footage.mp4 -o out.mp4 "
          "--weights models/visdrone.pt")


if __name__ == "__main__":
    main()
