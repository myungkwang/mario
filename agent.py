import random

import torch
import torch.nn.functional as F


class DQNAgent:
    """
    행동 선택과 DQN 학습을 담당하는 AI 선수.

    역할은 두 가지다.
    1. 지금 화면을 보고 어떤 버튼을 누를지 고른다.
    2. 저장된 경험을 보고 DQN 두뇌를 조금씩 고친다.
    """

    def __init__(
        self,
        online_net: torch.nn.Module,
        target_net: torch.nn.Module,
        num_actions: int,
        device: torch.device,
        gamma: float = 0.99,
    ):
        # online_net은 매번 학습으로 바뀌는 "현재 두뇌"다.
        # GPU가 있으면 .to(device)로 GPU에 올.려 계산을 빠르게 한다.
        self.online_net = online_net.to(device)

        # target_net은 정답을 계산할 때 쓰는 "천천히 바뀌는 두뇌"다.
        # DQN은 정답도 AI가 만들기 때문에, 정답 만드는 두뇌가 너무 빨리 바뀌면 흔들린다.
        self.target_net = target_net.to(device)

        # 행동 개수. COMPLEX_MOVEMENT 환경에서는 12개다.
        self.num_actions = num_actions

        # device는 "CPU에서 계산할지, GPU에서 계산할지"를 담는 값이다.
        self.device = device

        # gamma는 미래 보상을 얼마나 중요하게 볼지 정한다.
        # 0.99면 당장 점수뿐 아니라 조금 뒤 받을 점수도 중요하게 본다.
        self.gamma = gamma

        # target_net은 정답 계산용 복사본이다.
        # online_net만 계속 바꾸면 목표값도 흔들려서 학습이 불안정해진다.
        self.sync_target()

    def sync_target(self) -> None:
        # online_net의 현재 지식을 target_net에 복사한다.
        # 이 작업은 매 step마다 하지 않고 가끔만 한다.
        self.target_net.load_state_dict(self.online_net.state_dict())

        # target_net은 학습하지 않고 정답 계산만 하므로 eval 모드로 둔다.
        self.target_net.eval()

    def select_action(self, state: torch.Tensor, epsilon: float) -> int:
        # epsilon 확률로 랜덤 행동을 한다.
        # 처음부터 똑똑한 척만 하면 새 길을 못 찾으므로 일부러 탐험한다.
        #
        # 예: epsilon=1.0이면 항상 랜덤.
        # 예: epsilon=0.05이면 5%만 랜덤, 95%는 AI 판단.
        if random.random() < epsilon:
            return random.randrange(self.num_actions)

        # torch.no_grad()는 "지금은 학습하지 않고 행동만 고른다"는 뜻이다.
        # 이걸 쓰면 메모리를 덜 쓰고 계산도 조금 빨라진다.
        with torch.no_grad():
            # state 모양은 [4, 84, 84]다.
            # DQN은 batch 모양 [batch, 4, 84, 84]를 기대하므로 앞에 1칸을 추가한다.
            state = state.unsqueeze(0).to(self.device)

            # q_values는 행동별 예상 점수다.
            # 예: [0.1, 3.2, 1.5, -0.2, 2.0]
            q_values = self.online_net(state)

            # 가장 큰 Q값의 위치가 선택할 행동 번호다.
            return int(q_values.argmax(dim=1).item())

    def train_step(self, batch, optimizer: torch.optim.Optimizer) -> float:
        # replay buffer에서 꺼낸 batch를 GPU/CPU device로 옮긴다.
        # 모델과 데이터가 같은 device에 있어야 계산할 수 있다.
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        next_states = batch.next_states.to(self.device)
        dones = batch.dones.to(self.device)

        # 지금 상태에서 실제로 선택했던 행동의 Q값만 꺼낸다.
        #
        # self.online_net(states)는 모든 행동의 Q값을 준다.
        # gather는 그중 "실제로 눌렀던 행동"의 Q값만 골라낸다.
        # DQN은 실제 행동의 예상 점수를 정답에 가깝게 고치는 방식으로 배운다.
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: 다음 행동 선택은 online_net, 점수 평가는 target_net.
            # 한 네트워크가 혼자 고르고 평가하면 점수를 과대평가하기 쉽다.
            #
            # next_actions: 다음 화면에서 online_net이 제일 좋다고 고른 행동.
            next_actions = self.online_net(next_states).argmax(dim=1)

            # next_q: 그 행동이 target_net 기준으로 얼마나 좋은지 평가한 점수.
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # DQN의 핵심 공식:
            # 목표 점수 = 지금 받은 점수 + 미래 점수
            # done이면 게임이 끝났으므로 미래 점수는 0으로 만든다.
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # smooth_l1_loss는 예측값 current_q와 목표값 target_q의 차이를 잰다.
        # MSE보다 큰 실수에 덜 민감해서 DQN에서 자주 쓴다.
        loss = F.smooth_l1_loss(current_q, target_q)

        # 이전 학습에서 남아 있던 gradient를 지운다.
        # 지우지 않으면 gradient가 계속 더해져 이상하게 학습된다.
        optimizer.zero_grad()

        # loss를 줄이려면 각 숫자를 어느 방향으로 바꿔야 하는지 계산한다.
        loss.backward()

        # 큰 gradient가 튀면 학습이 망가질 수 있어 적당히 자른다.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)

        # optimizer가 계산된 gradient를 보고 online_net의 숫자들을 실제로 수정한다.
        optimizer.step()

        # train.py에서 TensorBoard에 기록하기 쉽게 Python 숫자로 돌려준다.
        return float(loss.item())
