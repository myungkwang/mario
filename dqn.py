import torch
from torch import nn


class DQN(nn.Module):
    """
    Mario 화면을 보고 "각 행동이 얼마나 좋은지" 점수로 말해주는 AI 두뇌.

    DQN에서 Q값은 "이 행동을 하면 앞으로 받을 점수의 예상값"이다.
    예를 들어 출력이 [1.0, 5.0, 2.0]이면 1번 행동이 가장 좋아 보인다는 뜻이다.
    """

    def __init__(self, input_channels: int, num_actions: int):
        # nn.Module 부모 초기화. PyTorch 모델을 만들 때 꼭 필요하다.
        # 이것을 해야 PyTorch가 내부 층과 학습할 숫자들을 추적할 수 있다.
        super().__init__()

        # Atari DQN에서 검증된 구조를 사용한다.
        # 게임 화면은 사진이라서, 일반 선형층보다 CNN이 벽/구멍/적 위치를 더 잘 본다.
        #
        # input_channels=4인 이유:
        # 사진 한 장만 보면 마리오가 움직이는 중인지 멈춘 중인지 알기 어렵다.
        # 그래서 최근 화면 4장을 같이 넣어 "움직임"까지 보게 한다.
        self.features = nn.Sequential(
            # Conv2d는 이미지에서 중요한 모양을 찾는 돋보기 역할이다.
            # kernel_size=8은 8x8 크기로 화면을 훑고, stride=4는 4칸씩 건너뛴다.
            # 이렇게 하면 화면 크기가 줄어 계산이 빨라진다.
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),

            # ReLU는 음수 값을 0으로 만든다.
            # 복잡한 판단을 배우기 위해 중간중간 넣는 기본 활성화 함수다.
            nn.ReLU(),

            # 두 번째 Conv2d는 첫 번째가 찾은 단순한 모양을 조합한다.
            # 예: 바닥, 벽, 적, 구멍 같은 더 큰 패턴.
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # 세 번째 Conv2d는 더 깊은 특징을 만든다.
            # 예: "앞에 적이 있고 점프해야 함" 같은 상황을 구분하는 재료가 된다.
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # 84x84 입력이 CNN을 지나면 64개 채널의 7x7 특징 지도가 된다.
        # 이것을 길게 펴서 "행동별 점수"로 바꾼다.
        self.q_head = nn.Sequential(
            # CNN 결과는 [채널, 세로, 가로] 모양이다.
            # Linear 층은 긴 줄 하나를 받아야 하므로 Flatten으로 펼친다.
            nn.Flatten(),

            # 64 * 7 * 7개 숫자를 512개 생각 공간으로 압축한다.
            # 512는 너무 작지도 크지도 않은 Atari DQN의 흔한 선택이다.
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),

            # 마지막 출력 개수는 행동 개수와 같아야 한다.
            # COMPLEX_MOVEMENT는 행동이 12개라서 Q값도 12개가 나온다.
            nn.Linear(512, num_actions),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # 원본 frame은 0~255 픽셀값이다. 0~1로 줄이면 학습이 더 안정적이다.
        # 사람으로 치면 너무 큰 숫자 대신 작은 비율로 바꿔서 계산을 쉽게 하는 것이다.
        frames = frames.float() / 255.0

        # features가 화면에서 특징을 찾고, q_head가 행동별 점수로 바꾼다.
        return self.q_head(self.features(frames))
