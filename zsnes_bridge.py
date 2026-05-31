from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class ActionSpec:
    name: str
    keys: frozenset[str]


ACTIONS = (
    ActionSpec("NOOP", frozenset()),
    ActionSpec("right", frozenset({"right"})),
    ActionSpec("right+A", frozenset({"right", "x"})),
    ActionSpec("right+B", frozenset({"right", "z"})),
    ActionSpec("right+A+B", frozenset({"right", "x", "z"})),
)


def parse_crop(value: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("crop must be x,y,w,h")

    try:
        x, y, width, height = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("crop values must be integers") from exc

    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("crop must use non-negative x/y and positive width/height")

    return x, y, width, height


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    if frame.size == 0:
        raise ValueError("frame is empty")

    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError("frame must be grayscale, BGR, or BGRA")

    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized.astype(np.uint8))


class FrameStacker:
    def __init__(self, size: int = 4):
        if size <= 0:
            raise ValueError("size must be positive")

        self.frames: deque[torch.Tensor] = deque(maxlen=size)
        self.size = size

    def reset(self, frame: torch.Tensor) -> torch.Tensor:
        self.frames.clear()
        for _ in range(self.size):
            self.frames.append(frame)
        return self.state()

    def push(self, frame: torch.Tensor) -> torch.Tensor:
        if not self.frames:
            return self.reset(frame)

        self.frames.append(frame)
        return self.state()

    def state(self) -> torch.Tensor:
        if len(self.frames) != self.size:
            raise RuntimeError("frame stack is not initialized")

        return torch.stack(tuple(self.frames), dim=0)


def build_key_plan(previous: set[str], current: set[str]) -> tuple[set[str], set[str]]:
    release_keys = previous - current
    press_keys = current - previous
    return release_keys, press_keys
