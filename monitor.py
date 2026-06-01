from pathlib import Path
import argparse
import time

import torch

from agent import DQNAgent
from dqn import DQN
from env import make_env


CHECKPOINT_PATH = Path("checkpoints/latest.pt")


def default_checkpoint_path(stage: str) -> Path:
    if stage == "1-1":
        return CHECKPOINT_PATH
    return Path("checkpoints") / f"{stage}_latest.pt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Watch the trained Mario DQN play.")
    parser.add_argument("--stage", default="1-1", help="Mario stage to watch, e.g. 1-1 or 1-2")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fps", type=int, default=60, help="Maximum render FPS")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Random action probability while watching. 0.0 shows the deterministic learned policy.",
    )
    return parser.parse_args(argv)


def load_agent(env, device: torch.device, checkpoint_path: Path = CHECKPOINT_PATH) -> DQNAgent:
    # monitor.py도 train.py와 같은 모양의 두뇌를 만들어야 한다.
    # 저장된 숫자만 불러오는 것이므로, 모델 구조가 같아야 한다.
    num_actions = env.action_space.n
    online_net = DQN(input_channels=4, num_actions=num_actions)
    target_net = DQN(input_channels=4, num_actions=num_actions)
    agent = DQNAgent(online_net, target_net, num_actions=num_actions, device=device)

    # map_location=device는 저장된 모델을 현재 컴퓨터의 GPU/CPU 위치에 맞게 불러온다.
    # 예: GPU에서 저장했지만 CPU에서 볼 수도 있게 해준다.
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # train.py가 저장한 "배운 지식"을 monitor의 두뇌에 넣는다.
    agent.online_net.load_state_dict(checkpoint["online_net"])
    agent.target_net.load_state_dict(checkpoint["target_net"])

    # monitor는 학습하지 않고 보기만 하므로 eval 모드로 둔다.
    agent.online_net.eval()
    return agent


def play_once(agent: DQNAgent, env, fps: int, epsilon: float = 0.0) -> float:
    # 한 판을 새로 시작한다.
    state = env.reset()
    done = False
    total_reward = 0.0
    frame_delay = 0.8 / fps

    while not done:
        started = time.perf_counter()

        # 기본 monitor는 학습이 아니라 관찰이므로 epsilon=0.0, 즉 현재 실력 그대로 본다.
        # 랜덤 행동을 섞으면 "AI가 진짜 배운 행동"을 보기 어렵다.
        # 다르게 움직이는지 보고 싶을 때만 --epsilon으로 탐험을 조금 섞는다.
        action = agent.select_action(state, epsilon=epsilon)

        # 행동을 게임에 넣고 다음 상태를 받는다.
        state, reward, done, info = env.step(action)
        total_reward += reward

        # env.render()가 실제 게임 화면 창을 보여준다.
        # train.py에서 이걸 켜면 학습이 느려져서 monitor.py에만 둔다.
        env.render()

        elapsed = time.perf_counter() - started
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

    return total_reward


def main(argv=None):
    args = parse_args(argv)
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not 0.0 <= args.epsilon <= 1.0:
        raise ValueError("--epsilon must be between 0.0 and 1.0")

    # GPU가 있으면 checkpoint를 GPU로 불러온다.
    # monitor는 학습보다 가벼워서 CPU여도 보통 볼 수 있다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 화면을 볼 목적이므로 render_mode="human"이라는 의도를 적어 둔다.
    # 실제 구형 gym-super-mario-bros는 env.render()로 창을 띄운다.
    env = make_env(render_mode="human", stage=args.stage)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(args.stage)

    print(f"monitor waiting for {checkpoint_path} stage={args.stage} fps={args.fps} epsilon={args.epsilon}")

    while True:
        # 학습이 아직 checkpoint를 저장하지 않았으면 기다린다.
        # train.py는 50000 step마다 checkpoints/latest.pt를 만든다.
        if not checkpoint_path.exists():
            time.sleep(5)
            continue

        # 최신 checkpoint를 불러와 현재 실력의 AI를 만든다.
        agent = load_agent(env, device, checkpoint_path=checkpoint_path)

        # 한 판 플레이를 화면으로 보여준다.
        reward = play_once(agent, env, fps=args.fps, epsilon=args.epsilon)
        print(f"episode reward={reward:.1f}")

        # 학습 파일이 다시 저장될 시간을 준다.
        # 너무 자주 불러오면 같은 모델만 반복해서 보게 된다.
        time.sleep(30)


if __name__ == "__main__":
    # 이 파일을 직접 실행했을 때만 monitor를 시작한다.
    main()
