from pathlib import Path
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from agent import DQNAgent
from dqn import DQN
from env import make_env
from replay_buffer import ReplayBuffer


CHECKPOINT_DIR = Path("checkpoints")
LATEST_PATH = CHECKPOINT_DIR / "latest.pt"


def default_checkpoint_path(stage: str) -> Path:
    if stage == "1-1":
        return LATEST_PATH
    return CHECKPOINT_DIR / f"{stage}_latest.pt"


def default_log_dir(stage: str) -> str:
    if stage == "1-1":
        return "runs/mario_dqn"
    return f"runs/mario_dqn_{stage}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Mario DQN on one stage.")
    parser.add_argument("--stage", default="1-1", help="Mario stage to train, e.g. 1-1 or 1-2")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to resume/save. Default: checkpoints/latest.pt for 1-1, checkpoints/<stage>_latest.pt otherwise.",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="Checkpoint to copy model weights from when --checkpoint does not exist yet.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="TensorBoard log directory. Default: runs/mario_dqn for 1-1, runs/mario_dqn_<stage> otherwise.",
    )
    return parser.parse_args(argv)


def epsilon_by_step(step: int, start: float = 1.0, end: float = 0.05, decay_steps: int = 1_000_000) -> float:
    # epsilon은 "랜덤 행동을 할 확률"이다.
    # 처음에는 아무것도 모르므로 1.0, 즉 100% 랜덤으로 많이 탐험한다.
    # 시간이 지나면 0.05까지 줄여서 5%만 탐험하고 95%는 배운 대로 움직인다.
    progress = min(step / decay_steps, 1.0)

    # start에서 end까지 천천히 직선으로 줄인다.
    # min을 쓰는 이유는 decay_steps를 넘으면 progress가 1보다 커지지 않게 하려는 것이다.
    return start + progress * (end - start)


def save_checkpoint(path: Path, agent: DQNAgent, optimizer: torch.optim.Optimizer, step: int, episode: int) -> None:
    # checkpoints 폴더가 없으면 만든다.
    # exist_ok=True라서 이미 있어도 에러가 나지 않는다.
    path.parent.mkdir(parents=True, exist_ok=True)

    # torch.save는 모델의 현재 상태를 파일로 저장한다.
    # 나중에 monitor.py가 이 파일을 읽어 현재 실력으로 플레이를 보여준다.
    torch.save(
        {
            # step과 episode는 "어디까지 학습했는지" 기록하는 표지판이다.
            "step": step,
            "episode": episode,

            # state_dict는 모델 안 숫자들, 즉 배운 지식이다.
            "online_net": agent.online_net.state_dict(),
            "target_net": agent.target_net.state_dict(),

            # optimizer 상태도 저장하면 나중에 이어서 학습할 때 도움이 된다.
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint_if_exists(
    path: Path,
    agent: DQNAgent,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, bool]:
    # 이어서 학습하려면 예전에 저장한 파일이 있는지 먼저 본다.
    # 파일이 없으면 처음 학습하는 것이므로 step=0, episode=0부터 시작한다.
    if not path.exists():
        return 0, 0, False

    # checkpoint는 게임 저장 파일과 비슷하다.
    # 마리오 게임에서 저장 파일을 불러오면 월드, 목숨, 위치가 돌아오듯이,
    # 여기서는 AI 두뇌 숫자, optimizer 상태, 몇 step까지 했는지를 불러온다.
    checkpoint = torch.load(path, map_location=device)

    # online_net은 실제로 행동을 고르는 현재 두뇌다.
    # 이 숫자를 불러와야 AI가 처음부터 다시 배우지 않고, 예전에 배운 실력에서 시작한다.
    agent.online_net.load_state_dict(checkpoint["online_net"])

    # target_net은 학습을 안정시키는 복사 두뇌다.
    # 이것도 같이 불러와야 이어서 학습할 때 계산 기준이 갑자기 바뀌지 않는다.
    agent.target_net.load_state_dict(checkpoint["target_net"])

    # optimizer는 "방금 전까지 어떤 방향으로 공부하고 있었는지" 기억하는 선생님이다.
    # 모델 숫자만 불러오고 optimizer를 새로 만들면, 선생님 기억은 사라진다.
    # 그래도 학습은 되지만, 진짜 이어서 학습하려면 optimizer 상태도 같이 불러오는 것이 좋다.
    optimizer.load_state_dict(checkpoint["optimizer"])

    # step은 지금까지 버튼을 몇 번 눌러보며 공부했는지다.
    # 이 값을 이어받아야 epsilon 같은 학습 스케줄도 예전 위치에서 계속 간다.
    step = int(checkpoint.get("step", 0))

    # episode는 몇 판을 플레이했는지다.
    # TensorBoard 그래프가 이어져 보이도록 episode 번호도 이어받는다.
    episode = int(checkpoint.get("episode", 0))

    return step, episode, True


def load_weights_from_checkpoint(path: Path, agent: DQNAgent, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    agent.online_net.load_state_dict(checkpoint["online_net"])
    agent.target_net.load_state_dict(checkpoint["target_net"])


def main(argv=None):
    args = parse_args(argv)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(args.stage)
    init_from_path = Path(args.init_from) if args.init_from else None
    log_dir = args.log_dir or default_log_dir(args.stage)

    # CUDA GPU가 있으면 GPU로 학습한다.
    # GPU가 없으면 CPU로도 실행되지만 훨씬 느리다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 마리오 게임 환경을 만든다.
    # env는 AI가 행동을 넣으면 다음 화면과 점수를 돌려주는 게임 기계다.
    env = make_env(stage=args.stage)

    # 환경이 가진 행동 개수를 읽는다.
    # RIGHT_ONLY면 보통 5개 행동이 있다.
    num_actions = env.action_space.n

    # online_net은 학습으로 계속 바뀌는 현재 두뇌다.
    online_net = DQN(input_channels=4, num_actions=num_actions)

    # target_net은 정답 계산을 안정시키기 위한 복사 두뇌다.
    target_net = DQN(input_channels=4, num_actions=num_actions)

    # agent는 두뇌들을 이용해서 행동도 고르고 학습도 한다.
    agent = DQNAgent(online_net, target_net, num_actions=num_actions, device=device)

    # Adam은 모델 숫자를 어떻게 고칠지 정하는 최적화 도구다.
    # lr=1e-4는 DQN에서 흔히 쓰는 작고 안정적인 학습률이다.
    optimizer = torch.optim.Adam(agent.online_net.parameters(), lr=1e-4)

    # 경험 10만 개를 저장하는 기억장을 만든다.
    # 너무 작으면 다양한 상황을 못 보고, 너무 크면 메모리를 많이 쓴다.
    replay = ReplayBuffer(capacity=100_000)

    # TensorBoard에 reward/loss 같은 기록을 남긴다.
    # 나중에 그래프로 학습이 좋아지는지 볼 수 있다.
    writer = SummaryWriter(log_dir=log_dir)

    # 이번 실행에서 추가로 공부할 step 수다.
    # 예전에는 total_steps가 "처음부터 끝까지 100만 step"이라는 뜻이었다.
    # 이제는 이어서 학습할 수 있으므로, python train.py를 실행할 때마다
    # 현재 저장된 위치에서 100만 step을 더 공부한다.
    additional_steps = 1_000_000

    # 처음 1만 step은 기억장 채우기 시간이다.
    # 경험이 너무 적을 때 바로 학습하면 같은 장면만 보고 배워서 불안정하다.
    learning_starts = 10_000

    # 한 번 학습할 때 경험 32개를 뽑는다.
    batch_size = 32

    # 매 step마다 학습하면 느리고 비슷한 데이터만 계속 본다.
    # 그래서 4 step마다 한 번 학습한다.
    train_every = 4

    # target_net은 1만 step마다 online_net을 복사한다.
    # 너무 자주 복사하면 target_net을 둔 의미가 줄어든다.
    target_sync_every = 10_000

    # 5만 step마다 최신 모델을 저장한다.
    # monitor.py가 이 파일을 읽어 화면으로 보여준다.
    checkpoint_every = 50_000

    # 저장된 모델이 있으면 이어서 학습한다.
    # 없으면 start_step=0, episode=0이므로 처음부터 학습한다.
    start_step, episode, resumed = load_checkpoint_if_exists(checkpoint_path, agent, optimizer, device)

    if resumed:
        print(f"resumed {checkpoint_path} stage={args.stage} step={start_step} episode={episode}")
    elif init_from_path is not None:
        load_weights_from_checkpoint(init_from_path, agent, device)
        agent.sync_target()
        print(f"initialized stage={args.stage} from {init_from_path}; starting new training")
    else:
        print(f"no checkpoint found for stage={args.stage}; starting new training")

    # 이번 실행이 끝날 step 번호다.
    # 예: 저장 파일이 step=1,000,000이면 end_step=2,000,000이 된다.
    end_step = start_step + additional_steps

    # 게임을 처음 시작하고 첫 상태를 받는다.
    # 주의: replay buffer는 checkpoint에 저장하지 않는다.
    # 그래서 이어서 학습해도 처음 1만 step은 새 경험을 다시 모으는 시간이다.
    # 이것은 "두뇌는 이어받지만, 연습장 노트는 새로 쓰는 것"과 비슷하다.
    state = env.reset()
    episode_reward = 0.0
    episode_raw_reward = 0.0
    episode_length = 0

    for step in range(start_step + 1, end_step + 1):
        # 지금 step에 맞는 탐험 확률을 계산한다.
        epsilon = epsilon_by_step(step)

        # agent가 현재 화면을 보고 행동 번호를 고른다.
        action = agent.select_action(state, epsilon)

        # 선택한 행동을 게임에 넣고, 다음 화면과 보상을 받는다.
        next_state, reward, done, info = env.step(action)

        # 지금 경험을 기억장에 저장한다.
        # 나중에 랜덤으로 꺼내 학습할 것이다.
        replay.push(state, action, reward, next_state, done)

        # 다음 반복에서는 next_state가 현재 state가 된다.
        state = next_state
        episode_reward += reward
        episode_raw_reward += float(info.get("raw_reward", reward))
        episode_length += 1

        if done:
            stage_clear = bool(info.get("flag_get", False))
            dead = int(info.get("life", 0)) < 2 and not stage_clear

            # 한 판이 끝나면 이번 판 점수와 최대 이동 거리 같은 값을 기록한다.
            writer.add_scalar("episode/reward", episode_reward, episode)
            writer.add_scalar("episode/raw_reward", episode_raw_reward, episode)
            writer.add_scalar("episode/x_pos", info.get("x_pos", 0), episode)
            writer.add_scalar("episode/flag_get", int(info.get("flag_get", False)), episode)
            writer.add_scalar("episode/life", info.get("life", 0), episode)
            writer.add_scalar("episode/length", episode_length, episode)
            writer.add_scalar("episode/death", int(dead), episode)
            writer.add_scalar("episode/stage_clear", int(stage_clear), episode)

            # 새 게임을 시작한다.
            state = env.reset()
            episode += 1
            episode_reward = 0.0
            episode_raw_reward = 0.0
            episode_length = 0

        if len(replay) >= learning_starts and step % train_every == 0:
            # 기억장이 충분히 찼고, 학습할 차례면 batch를 뽑는다.
            batch = replay.sample(batch_size)

            # 뽑은 경험으로 DQN을 한 번 학습시킨다.
            loss = agent.train_step(batch, optimizer)

            # loss는 "AI 예측이 목표와 얼마나 다른지"다. 작아지는지 관찰한다.
            writer.add_scalar("train/loss", loss, step)

        if step % target_sync_every == 0:
            # target_net을 현재 online_net으로 갱신한다.
            agent.sync_target()

        if step % checkpoint_every == 0:
            # 최신 모델 저장. monitor.py가 이 파일로 화면을 보여준다.
            save_checkpoint(checkpoint_path, agent, optimizer, step, episode)
            writer.add_scalar("train/epsilon", epsilon, step)
            print(f"saved {checkpoint_path} stage={args.stage} step={step} episode={episode} epsilon={epsilon:.3f}")

    # 학습이 끝났을 때 마지막 상태도 저장한다.
    save_checkpoint(checkpoint_path, agent, optimizer, end_step, episode)

    # 게임과 TensorBoard writer를 정리한다.
    env.close()
    writer.close()


if __name__ == "__main__":
    # 이 파일을 직접 실행했을 때만 main()을 돌린다.
    # 다른 파일에서 import할 때 학습이 갑자기 시작되지 않게 막는 안전장치다.
    main()
