"""ONNX lane inference with Jetson CUDA sidecar and safe CPU fallback."""

from pathlib import Path
import platform
import select
import subprocess

import cv2
import numpy as np


class LaneNeural:
    INPUT_WIDTH = 320
    INPUT_HEIGHT = 180
    OUTPUT_BYTES = INPUT_WIDTH * INPUT_HEIGHT

    def __init__(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.is_file():
            raise FileNotFoundError(f"Lane model missing: {path}")
        self.path = path
        self.net = None
        self.worker = None
        self.backend = "cpu"
        if not self._start_cuda_sidecar():
            self._start_local_net()

    def _start_cuda_sidecar(self):
        if platform.machine().lower() not in ("aarch64", "arm64"):
            return False
        worker_path = Path(__file__).with_name("lane_gpu_worker.py")
        native_worker = Path(__file__).with_name("lane_gpu_worker")
        engine_path = self.path.with_name("lane_fp16.engine")
        system_python = Path("/usr/bin/python3")
        if native_worker.is_file() and engine_path.is_file():
            command = [str(native_worker), str(engine_path)]
        elif worker_path.is_file() and system_python.is_file():
            command = [str(system_python), str(worker_path), str(self.path)]
        else:
            return False
        try:
            self.worker = subprocess.Popen(
                command,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, bufsize=0)
                                                                               
                                                                             
                                                                             
            ready, _, _ = select.select([self.worker.stdout], [], [], 60.0)
            marker = self.worker.stdout.read(1) if ready else b""
            if marker != b"G":
                self._stop_worker()
                return False
            self.backend = "cuda_fp16_sidecar"
            return True
        except (OSError, subprocess.SubprocessError):
            self._stop_worker()
            return False

    def _start_local_net(self):
        self.net = cv2.dnn.readNetFromONNX(str(self.path))
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        except (AttributeError, cv2.error):
            cuda_devices = 0
        if cuda_devices > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            self.backend = "cuda_fp16"
        else:
            self.backend = "cpu"

    @staticmethod
    def _read_exact(stream, size):
        chunks = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _predict_sidecar(self, frame):
        resized = cv2.resize(
            frame, (self.INPUT_WIDTH, self.INPUT_HEIGHT),
            interpolation=cv2.INTER_AREA)
        self.worker.stdin.write(
            np.ascontiguousarray(resized, dtype=np.uint8).tobytes())
        self.worker.stdin.flush()
        ready, _, _ = select.select([self.worker.stdout], [], [], 3.0)
        if not ready:
            raise RuntimeError("CUDA lane worker timed out")
        raw = self._read_exact(self.worker.stdout, self.OUTPUT_BYTES)
        if len(raw) != self.OUTPUT_BYTES:
            raise RuntimeError("CUDA lane worker stopped")
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            self.INPUT_HEIGHT, self.INPUT_WIDTH)

    def predict(self, frame):
        if self.worker is not None:
            try:
                labels = self._predict_sidecar(frame)
                return cv2.resize(
                    labels, (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
            except (BrokenPipeError, OSError, RuntimeError):
                self._stop_worker()
                self._start_local_net()

        self.net.setInput(cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.INPUT_WIDTH, self.INPUT_HEIGHT),
            swapRB=True, crop=False))
        output = self.net.forward()
        if (output.shape != (1, 3, self.INPUT_HEIGHT, self.INPUT_WIDTH) or
                not np.isfinite(output).all()):
            raise ValueError("Invalid lane logits")
        labels = output[0].argmax(0).astype(np.uint8)
        return cv2.resize(
            labels, (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST)

    def _stop_worker(self):
        worker, self.worker = self.worker, None
        if worker is None:
            return
        for stream in (worker.stdin, worker.stdout):
            try:
                stream.close()
            except Exception:
                pass
        try:
            worker.terminate()
            worker.wait(timeout=1.0)
        except Exception:
            try:
                worker.kill()
            except Exception:
                pass

    def close(self):
        self._stop_worker()

    def __del__(self):
        self._stop_worker()
