"""Python 3.6-compatible CUDA/FP16 ONNX worker for Jetson system OpenCV."""

import sys

import cv2
import numpy as np


WIDTH = 320
HEIGHT = 180
FRAME_BYTES = WIDTH * HEIGHT * 3


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main():
    if len(sys.argv) != 2:
        return 2
    if not hasattr(cv2, "cuda") or cv2.cuda.getCudaEnabledDeviceCount() < 1:
        return 3
    net = cv2.dnn.readNetFromONNX(sys.argv[1])
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    source = sys.stdin.buffer
    sink = sys.stdout.buffer
    sink.write(b"G")
    sink.flush()
    while True:
        raw = read_exact(source, FRAME_BYTES)
        if not raw:
            break
        if len(raw) != FRAME_BYTES:
            return 4
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        blob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255.0, (WIDTH, HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        output = net.forward()
        if output.shape != (1, 3, HEIGHT, WIDTH):
            return 5
        labels = output[0].argmax(0).astype(np.uint8)
        sink.write(labels.tobytes())
        sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
