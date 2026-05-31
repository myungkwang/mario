import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from agent import DQNAgent
from dqn import DQN
from zsnes_bridge import ACTIONS, FrameStacker, build_key_plan, parse_crop, preprocess_frame


DEFAULT_EXE = r"E:\game\marioall\ZSNESW.EXE"
DEFAULT_CHECKPOINT = "checkpoints/latest.pt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Play ZSNES with the trained Mario DQN checkpoint.")
    parser.add_argument("--exe", default=DEFAULT_EXE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--crop", required=True, help="Crop inside emulator window as x,y,w,h")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--title", default="ZSNES", help="Window title substring")
    parser.add_argument(
        "--test-action",
        type=int,
        default=None,
        help="Force one RIGHT_ONLY action index for input debugging",
    )
    return parser.parse_args(argv)


def require_runtime_dependencies():
    missing = []
    for package in ("mss", "pygetwindow", "pynput"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        names = " ".join(missing)
        raise RuntimeError(f"missing runtime packages; install with: pip install {names}")


def validate_checkpoint(path: str) -> Path:
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    return checkpoint


def find_window(title_substring: str):
    import pygetwindow as gw

    candidates = [
        window
        for window in gw.getAllWindows()
        if title_substring.lower() in window.title.lower()
    ]
    visible = [window for window in candidates if window.width > 0 and window.height > 0]
    if not visible:
        raise RuntimeError(f"ZSNES window not found with title containing: {title_substring}")

    return visible[0]


def focus_window(window):
    try:
        window.activate()
        time.sleep(0.05)
    except Exception as exc:
        print(f"warning: could not focus ZSNES window: {exc}")


def capture_window(sct, window, crop: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = crop
    monitor = {
        "left": int(window.left + x),
        "top": int(window.top + y),
        "width": int(width),
        "height": int(height),
    }
    image = np.array(sct.grab(monitor))
    if image.size == 0:
        raise RuntimeError("captured empty frame")

    return image


def load_agent(checkpoint_path: Path, device: torch.device) -> DQNAgent:
    num_actions = len(ACTIONS)
    online_net = DQN(input_channels=4, num_actions=num_actions)
    target_net = DQN(input_channels=4, num_actions=num_actions)
    agent = DQNAgent(online_net, target_net, num_actions=num_actions, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.online_net.load_state_dict(checkpoint["online_net"])
    agent.target_net.load_state_dict(checkpoint["target_net"])
    agent.online_net.eval()
    return agent


class KeyboardController:
    def __init__(self):
        from pynput.keyboard import Controller, Key

        self.controller = Controller()
        self.key_map = {
            "right": Key.right,
            "x": "x",
            "z": "z",
        }
        self.held: set[str] = set()

    def apply(self, keys: set[str]):
        release_keys, press_keys = build_key_plan(self.held, keys)
        for key in release_keys:
            self.controller.release(self.key_map[key])
        for key in press_keys:
            self.controller.press(self.key_map[key])
        self.held = set(keys)

    def release_all(self):
        self.apply(set())


def draw_debug(crop_frame: np.ndarray, processed: torch.Tensor, action_name: str, fps: float):
    preview = crop_frame
    if preview.ndim == 3 and preview.shape[2] == 4:
        preview = cv2.cvtColor(preview, cv2.COLOR_BGRA2BGR)

    small = cv2.cvtColor(processed.numpy(), cv2.COLOR_GRAY2BGR)
    small = cv2.resize(small, (168, 168), interpolation=cv2.INTER_NEAREST)
    cv2.putText(
        preview,
        f"action={action_name} fps={fps:.1f}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imshow("ZSNES DQN crop", preview)
    cv2.imshow("ZSNES DQN 84x84", small)


def run(args):
    require_runtime_dependencies()
    checkpoint_path = validate_checkpoint(args.checkpoint)
    crop = parse_crop(args.crop)
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.test_action is not None and not 0 <= args.test_action < len(ACTIONS):
        raise ValueError(f"--test-action must be between 0 and {len(ACTIONS) - 1}")

    import mss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_agent(checkpoint_path, device)
    stacker = FrameStacker(size=4)
    keyboard = KeyboardController()
    window = find_window(args.title)
    target_delay = 1.0 / args.fps
    last_time = time.perf_counter()

    print(f"window={window.title!r} crop={crop} dry_run={args.dry_run} debug={args.debug}")
    print("press q in debug window or Ctrl+C in terminal to stop")

    try:
        with mss.mss() as sct:
            while True:
                started = time.perf_counter()
                crop_frame = capture_window(sct, window, crop)
                processed = preprocess_frame(crop_frame)
                state = stacker.push(processed)
                if args.test_action is None:
                    action = agent.select_action(state, epsilon=0.0)
                else:
                    action = args.test_action
                if action < 0 or action >= len(ACTIONS):
                    raise RuntimeError(f"model returned invalid action: {action}")

                action_spec = ACTIONS[action]
                if not args.dry_run:
                    focus_window(window)
                    keyboard.apply(set(action_spec.keys))

                now = time.perf_counter()
                fps = 1.0 / max(now - last_time, 1e-6)
                last_time = now
                print(f"action={action}:{action_spec.name} fps={fps:.1f}", end="\r")

                if args.debug:
                    draw_debug(crop_frame, processed, action_spec.name, fps)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                elapsed = time.perf_counter() - started
                if elapsed < target_delay:
                    time.sleep(target_delay - elapsed)
    finally:
        keyboard.release_all()
        if args.debug:
            cv2.destroyAllWindows()
        print()


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
