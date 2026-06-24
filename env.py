from collections import deque

import cv2
import gym
import gym_super_mario_bros
import numpy as np
import torch
from gym import spaces
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace


class MarioPreprocessWrapper(gym.Wrapper):
    """
    Mario 화면을 DQN이 먹기 좋은 4장짜리 흑백 이미지로 바꾼다.

    원래 게임 화면은 크고 색도 많다.
    AI가 처음부터 큰 컬러 화면을 보면 계산이 느리고 배우기 어렵다.
    그래서 화면을 작게 만들고, 흑백으로 바꾸고, 최근 4장을 묶는다.
    """

    def __init__(self, env, frame_stack: int = 4, frame_skip: int = 4, episodic_life: bool = True):
        # gym.Wrapper는 기존 env를 감싸서 행동/화면을 바꿔주는 포장지다.
        # 원래 Mario env는 그대로 두고, 관찰값만 DQN용으로 바꾼다.
        super().__init__(env)

        # 몇 장의 화면을 쌓을지 정한다.
        # 4장을 쓰면 마리오가 어느 방향으로 움직이는지 알 수 있다.
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.last_x_pos = 0
        self.last_life = 2

        # episodic_life=True면 목숨이 줄 때마다(죽을 때마다) 학습상 done=True로 본다.
        # 그러면 학습 루프가 바로 reset하므로, 죽음 애니메이션 동안의 쓸모없는 화면을
        # 에이전트가 보지 않고, 죽음 벌점도 제대로 terminal 신호로 들어간다.
        # monitor처럼 죽어도 남은 목숨으로 계속 이어보고 싶을 때는 False로 끈다.
        self.episodic_life = episodic_life

        # deque는 오래된 화면을 자동으로 밀어내는 줄이다.
        # maxlen=4이면 새 화면이 들어올 때 가장 오래된 화면이 빠진다.
        self.frames = deque(maxlen=frame_stack)

        # DQN 입력 모양: 최근 화면 4장, 각 화면은 84x84 흑백.
        # observation_space는 "AI가 받는 화면의 모양과 값 범위"를 gym에 알려준다.
        # low=0, high=255는 픽셀값 범위다.
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(frame_stack, 84, 84),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        # reset은 새 게임을 시작할 때 호출된다.
        # gym 버전에 따라 반환값이 obs 또는 (obs, info)일 수 있어 둘 다 처리한다.
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result

        # 원본 컬러 화면을 DQN용 84x84 흑백 화면으로 바꾼다.
        frame = self._preprocess(obs)

        # 게임 시작 직후에는 이전 화면이 없으므로 같은 화면 4장을 채운다.
        # 이렇게 해야 첫 state도 항상 [4, 84, 84] 모양을 가진다.
        for _ in range(self.frame_stack):
            self.frames.append(frame)

        self.last_x_pos = 0
        self.last_life = 2
        return self._get_state()

    def step(self, action):
        # action은 "몇 번째 버튼 조합을 누를지"를 뜻하는 숫자다.
        # env.step(action)은 게임을 한 칸 진행하고 결과를 돌려준다.
        total_raw_reward = 0.0
        done = False
        info = {}
        obs = None

        for _ in range(self.frame_skip):
            result = self.env.step(action)

            # gym 0.25는 4개, gym 0.26은 5개를 돌려준다. 둘 다 받아준다.
            # 이 코드를 넣으면 라이브러리 버전 차이 때문에 바로 깨지는 일을 줄일 수 있다.
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result

            total_raw_reward += float(reward)
            if done:
                break

        # _shape_reward가 self.last_life를 갱신하기 전에 죽음 여부를 잡아둔다.
        prev_life = self.last_life
        shaped_reward = self._shape_reward(total_raw_reward, info, done)
        life_lost = int(info.get("life", prev_life)) < prev_life

        if obs is not None:
            # 새 화면도 똑같이 작고 흑백인 frame으로 바꿔서 frames에 넣는다.
            self.frames.append(self._preprocess(obs))

        # episodic_life: 죽으면(목숨 감소) 남은 목숨이 있어도 학습상 done=True로 본다.
        # 학습 루프가 곧바로 reset → 깨끗한 (랜덤) 스테이지 시작.
        # 덕분에 죽음 애니메이션 화면을 에이전트가 보지 않고, 죽음 벌점이 terminal로 정확히 들어간다.
        if self.episodic_life and life_lost:
            done = True

        # DQN 학습 루프가 쓰기 쉬운 형태로 돌려준다.
        # state, reward, done, info는 강화학습의 기본 네 가지 정보다.
        info = dict(info)
        info["raw_reward"] = total_raw_reward
        info["shaped_reward"] = shaped_reward
        return self._get_state(), shaped_reward, bool(done), info

    def _shape_reward(self, raw_reward: float, info: dict, done: bool) -> float:
        # gym_super_mario_bros raw_reward는 이미 전진속도 보상 + 시계 패널티를 담고 있다.
        # 그래서 전진을 또 보상하지 않고(이중보상 제거), 정지 벌점도 두지 않는다.
        # 정지 벌점이 있으면 피라냐 파이프 앞에서 "잠깐 기다리기"를 못 배워 돌진하다 죽는다.
        x_pos = int(info.get("x_pos", self.last_x_pos))
        life = int(info.get("life", self.last_life))
        flag_get = bool(info.get("flag_get", False))

        # raw_reward는 env step당 대략 [-15, 15]. /10으로 줄여 목표 Q값을 작게 유지한다.
        # 이래야 깃발/죽음 보너스가 묻히지 않고 학습이 안정적이다.
        reward = raw_reward / 10.0

        if flag_get:
            # 클리어 보너스. 1000은 너무 커서 Q값을 흔들었다 -> 다른 보상과 자릿수를 맞춘다.
            reward += 50.0
        elif life < self.last_life:
            # 목숨이 줄면 죽은 것. done 여부와 무관하게 벌점을 준다.
            # 전체 게임(SuperMarioBros-v0)은 목숨이 남으면 죽어도 에피소드가 안 끝나므로,
            # done만 보면 중간 죽음을 놓친다. 적별 보상 손코딩 없이도 에이전트는
            # "죽으면 손해"를 배워 모든 월드에서 회피(후진/점프)를 학습한다.
            reward -= 15.0

        self.last_x_pos = max(self.last_x_pos, x_pos)
        self.last_life = life
        return float(reward)

    def _preprocess(self, obs):
        # RGB 화면을 흑백으로 바꾸면 색 정보는 줄지만 계산이 빨라진다.
        # 마리오는 색보다 "벽, 적, 구멍, 위치"가 더 중요해서 흑백으로도 충분히 시작 가능하다.
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # 84x84는 Atari DQN에서 많이 쓰는 표준 크기다.
        # 작아져서 계산은 빨라지고, 중요한 구조는 대체로 남는다.
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        # uint8은 0~255 픽셀을 작게 저장하는 타입이다.
        # replay buffer에 많이 저장해야 하므로 메모리를 아끼는 게 중요하다.
        return resized.astype(np.uint8)

    def _get_state(self):
        # frames에는 84x84 화면 4장이 들어 있다.
        # np.stack으로 [4, 84, 84] 한 덩어리로 만든 뒤 torch Tensor로 바꾼다.
        return torch.from_numpy(np.stack(self.frames, axis=0))


def make_env(render_mode: str | None = None, stage: str = "1-1", episodic_life: bool = True):
    """
    선택한 Mario 스테이지 환경 생성.

    COMPLEX_MOVEMENT로 행동을 12개 쓴다.
    left(후진)·left+A(뒤로 점프)로 적을 피하고, A(점프)로 뛰어넘고,
    무엇보다 down으로 파이프 안에 들어갈 수 있다.
    전체 게임은 파이프로 내려가야 완료되는 스테이지가 있어 down이 필수다.

    stage="all"이면 SuperMarioBros-v0(전체 게임) env를 만든다.
    이 env는 1-1부터 시작해 깃발을 닿으면 자동으로 다음 스테이지로 넘어가고,
    목숨을 다 잃거나 게임을 클리어할 때까지 한 에피소드로 이어진다.
    하나의 모델로 1-1~8-8을 연속 플레이/학습하려면 이걸 쓴다.

    stage="random"이면 SuperMarioBrosRandomStages-v0 env를 만든다.
    매 reset마다 랜덤 스테이지에서 시작해 전 스테이지를 골고루 학습한다.
    "all"은 항상 1-1부터라 후반 스테이지는 거의 못 가서 학습이 안 된다.
    그래서 학습은 "random"으로 골고루, 연속 플레이 감상은 "all"로 보는 게 좋다.
    """

    if stage in ("all", "full", "fullgame"):
        # 스테이지 suffix 없는 id = 전체 게임. 스테이지가 자동으로 이어진다.
        env = gym_super_mario_bros.make("SuperMarioBros-v0")
    elif stage in ("random", "rand"):
        # 매 reset마다 랜덤 스테이지. 전 스테이지 균등 노출로 후반도 학습된다.
        env = gym_super_mario_bros.make("SuperMarioBrosRandomStages-v0")
    else:
        # SuperMarioBros-1-1-v0, SuperMarioBros-1-2-v0처럼 스테이지 하나만 연습한다.
        env = gym_super_mario_bros.make(f"SuperMarioBros-{stage}-v0")

    # JoypadSpace는 복잡한 버튼 조합을 몇 개 행동으로 줄여준다.
    # COMPLEX_MOVEMENT(12개): NOOP, right, right+A, right+B, right+A+B,
    #   A(점프), left, left+A, left+B, left+A+B, down(파이프 진입), up.
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    # render_mode 인자는 monitor.py에서 의도를 드러내기 위한 자리다.
    # 이 라이브러리는 구형 gym API라 생성자 render_mode를 지원하지 않는다.
    _ = render_mode

    # 학습은 episodic_life=True(죽을 때마다 깔끔히 reset), monitor는 False로 호출해
    # 죽어도 남은 목숨으로 1-1~8-8 연속 플레이를 이어서 본다.
    return MarioPreprocessWrapper(env, episodic_life=episodic_life)
