from collections import deque

import cv2
import gym
import gym_super_mario_bros
import numpy as np
import torch
from gym import spaces
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace


class MarioPreprocessWrapper(gym.Wrapper):
    """
    Mario 화면을 DQN이 먹기 좋은 4장짜리 흑백 이미지로 바꾼다.

    원래 게임 화면은 크고 색도 많다.
    AI가 처음부터 큰 컬러 화면을 보면 계산이 느리고 배우기 어렵다.
    그래서 화면을 작게 만들고, 흑백으로 바꾸고, 최근 4장을 묶는다.
    """

    def __init__(self, env, frame_stack: int = 4, frame_skip: int = 4):
        # gym.Wrapper는 기존 env를 감싸서 행동/화면을 바꿔주는 포장지다.
        # 원래 Mario env는 그대로 두고, 관찰값만 DQN용으로 바꾼다.
        super().__init__(env)

        # 몇 장의 화면을 쌓을지 정한다.
        # 4장을 쓰면 마리오가 어느 방향으로 움직이는지 알 수 있다.
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.last_x_pos = 0
        self.last_life = 2

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

        shaped_reward = self._shape_reward(total_raw_reward, info, done)

        if obs is not None:
            # 새 화면도 똑같이 작고 흑백인 frame으로 바꿔서 frames에 넣는다.
            self.frames.append(self._preprocess(obs))

        # DQN 학습 루프가 쓰기 쉬운 형태로 돌려준다.
        # state, reward, done, info는 강화학습의 기본 네 가지 정보다.
        info = dict(info)
        info["raw_reward"] = total_raw_reward
        info["shaped_reward"] = shaped_reward
        return self._get_state(), shaped_reward, bool(done), info

    def _shape_reward(self, raw_reward: float, info: dict, done: bool) -> float:
        x_pos = int(info.get("x_pos", self.last_x_pos))
        life = int(info.get("life", self.last_life))
        flag_get = bool(info.get("flag_get", False))

        progress = max(0, x_pos - self.last_x_pos)
        reward = raw_reward + progress * 0.1

        if progress == 0:
            reward -= 0.2
        if done and life < self.last_life and not flag_get:
            reward -= 50.0
        if flag_get:
            reward += 1000.0

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


def make_env(render_mode: str | None = None, stage: str = "1-1"):
    """
    선택한 Mario 스테이지 환경 생성.

    처음부터 모든 버튼을 쓰게 하면 경우의 수가 너무 많아 배움이 느리다.
    그래서 처음 학습은 오른쪽 위주 행동만 있는 RIGHT_ONLY로 작게 시작한다.
    """

    # SuperMarioBros-1-1-v0, SuperMarioBros-1-2-v0처럼 스테이지 하나를 골라 연습한다.
    env = gym_super_mario_bros.make(f"SuperMarioBros-{stage}-v0")

    # JoypadSpace는 복잡한 버튼 조합을 몇 개 행동으로 줄여준다.
    # RIGHT_ONLY에는 오른쪽 이동, 오른쪽 점프 같은 초보 학습용 행동만 있다.
    env = JoypadSpace(env, RIGHT_ONLY)

    # render_mode 인자는 monitor.py에서 의도를 드러내기 위한 자리다.
    # 이 라이브러리는 구형 gym API라 생성자 render_mode를 지원하지 않는다.
    _ = render_mode

    return MarioPreprocessWrapper(env)
