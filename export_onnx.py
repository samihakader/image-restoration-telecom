"""Re-export a trained model.pt to ONNX.

    python export_onnx.py --ckpt models/model.pt --out models/model.onnx
"""
import argparse

from restoration.model import export_onnx, load_checkpoint, verify_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/model.pt")
    ap.add_argument("--out", default="models/model.onnx")
    args = ap.parse_args()
    model = load_checkpoint(args.ckpt)
    export_onnx(model, args.out)
    print("exported", args.out, "| max diff vs PyTorch:", verify_onnx(model, args.out))


if __name__ == "__main__":
    main()
