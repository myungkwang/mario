from collections import deque
from dataclasses import dataclass
import random

import torch


@dataclass
class ReplayBatch:
    # dataclass를 쓰면 묶음 상자를 쉽게 만들 수 있다.
    # states, actions 같은 값을 이름으로 꺼낼 수 있어 코드가 읽기 쉬워진다.
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """
    Mario가 겪은 일을 저장하는 기억장.

    DQN은 바로 직전 일만 보고 배우면 안 좋다.
    예를 들어 계속 오른쪽만 가던 기억만 보면 점프가 필요한 순간을 잘 못 배운다.
    그래서 많은 경험을 저장하고, 그중 일부를 랜덤으로 뽑아 골고루 공부한다.
    """

    def __init__(self, capacity: int):
        # deque(maxlen=...)은 꽉 차면 오래된 기억부터 자동 삭제한다.
        # DQN은 최근 경험만 쓰면 편향되므로, 여러 기억에서 랜덤으로 뽑아 학습한다.
        # capacity=100000이면 경험 10만 개까지만 보관한다.
        # 무한히 저장하면 메모리가 계속 커지기 때문이다.
        self.memory = deque(maxlen=capacity)

    def __len__(self) -> int:
        # len(replay)를 쓸 수 있게 해준다.
        # train.py에서 "기억이 충분히 쌓였나?" 확인할 때 쓴다.
        return len(self.memory)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        # 경험 하나는 이렇게 생겼다:
        # 현재 화면, 누른 버튼, 받은 점수, 다음 화면, 게임 끝났는지 여부.
        # 이것이 강화학습의 기본 공부 자료다.
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> ReplayBatch:
        # random.sample은 기억장에서 서로 다른 경험을 batch_size개 뽑는다.
        # 순서대로 뽑지 않는 이유는 비슷한 장면만 연속 학습하면 AI가 한쪽으로 치우치기 때문이다.
        samples = random.sample(self.memory, batch_size)

        # zip(*samples)는 경험 묶음을 종류별로 다시 나눈다.
        # 예: [(s1,a1,r1), (s2,a2,r2)] -> (s1,s2), (a1,a2), (r1,r2)
        states, actions, rewards, next_states, dones = zip(*samples)

        return ReplayBatch(
            # torch.stack은 여러 장의 state를 하나의 큰 텐서로 쌓는다.
            # 결과 모양은 [batch, 4, 84, 84]가 된다.
            states=torch.stack(states),

            # action은 행동 번호라서 정수 long 타입이 필요하다.
            # 나중에 gather로 Q값을 꺼낼 때 long 타입이어야 한다.
            actions=torch.tensor(actions, dtype=torch.long),

            # reward는 점수 계산에 쓰므로 float 타입으로 만든다.
            rewards=torch.tensor(rewards, dtype=torch.float32),
            next_states=torch.stack(next_states),

            # done은 True/False지만 계산할 때 1.0/0.0 숫자로 쓰기 편하다.
            dones=torch.tensor(dones, dtype=torch.float32),
        )
