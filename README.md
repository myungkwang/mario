# Mario DQN

`gym-super-mario-bros` 환경에서 슈퍼 마리오 1-1 스테이지를 DQN으로 학습시키는 Python 프로젝트입니다.  
최근 4프레임을 84x84 흑백 이미지로 전처리하고, Double DQN 방식으로 행동별 Q값을 학습합니다.

## 주요 기능

- `SuperMarioBros-1-1-v0` 환경 학습
- `RIGHT_ONLY` 행동 공간 사용
- 프레임 스킵, 프레임 스택, 보상 shaping 적용
- replay buffer 기반 DQN 학습
- `checkpoints/latest.pt` 자동 저장 및 이어서 학습
- TensorBoard 학습 로그 기록
- 학습된 checkpoint를 화면으로 확인하는 모니터 실행
- ZSNES 창 화면을 캡처해 학습 모델로 실제 에뮬레이터 입력 제어

## 프로젝트 구조

```text
.
├── agent.py              # 행동 선택과 DQN 학습 로직
├── dqn.py                # CNN 기반 Q-network
├── env.py                # Mario 환경 생성 및 전처리 wrapper
├── replay_buffer.py      # replay buffer 구현
├── train.py              # 학습 진입점
├── monitor.py            # checkpoint 플레이 화면 확인
├── play_zsnes.py         # ZSNES 창 제어 실행기
├── zsnes_bridge.py       # ZSNES 캡처/전처리/키 입력 보조 로직
├── tests/                # 단위 테스트
├── checkpoints/          # 학습 checkpoint 저장 위치
└── runs/                 # TensorBoard 로그 저장 위치
```

## 설치

Python 3.11 기준으로 개발되었습니다.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch gym gym-super-mario-bros nes-py opencv-python numpy tensorboard
```

ZSNES 브릿지를 사용할 경우 추가 패키지가 필요합니다.

```bash
pip install mss pygetwindow pynput
```

## 학습 실행

```bash
python train.py
```

학습은 기본적으로 현재 checkpoint 이후 1,000,000 step을 추가로 진행합니다.

- checkpoint 위치: `checkpoints/latest.pt`
- 저장 주기: 50,000 step
- TensorBoard 로그: `runs/mario_dqn`
- replay buffer는 checkpoint에 저장하지 않으므로 재시작 후 초반 10,000 step은 새 경험을 다시 모읍니다.

## 학습 로그 확인

```bash
tensorboard --logdir runs
```

브라우저에서 TensorBoard 주소를 열면 episode reward, raw reward, x position, loss, epsilon 값을 확인할 수 있습니다.

## 학습 모델 화면 확인

```bash
python monitor.py
```

옵션:

```bash
python monitor.py --checkpoint checkpoints/latest.pt --fps 60 --epsilon 0.0
```

- `--checkpoint`: 불러올 checkpoint 경로
- `--fps`: 화면 렌더링 최대 FPS
- `--epsilon`: 관찰 중 랜덤 행동 확률. `0.0`이면 학습된 정책 그대로 실행합니다.

## ZSNES에서 실행

학습된 checkpoint를 ZSNES 창에 연결해 실제 키 입력을 보낼 수 있습니다.

```bash
python play_zsnes.py --crop x,y,w,h
```

예시:

```bash
python play_zsnes.py --crop 32,48,512,448 --fps 10 --debug
```

주요 옵션:

- `--exe`: ZSNES 실행 파일 경로
- `--checkpoint`: 사용할 checkpoint 경로
- `--crop`: 에뮬레이터 창 안에서 게임 화면을 잘라낼 영역. `x,y,w,h` 형식
- `--fps`: 입력/캡처 루프 FPS
- `--debug`: 캡처 화면과 84x84 전처리 화면 표시
- `--dry-run`: 키 입력 없이 행동 추론만 실행
- `--title`: 찾을 창 제목 일부. 기본값은 `ZSNES`
- `--test-action`: 모델 대신 특정 행동 번호를 강제로 입력

기본 행동 매핑:

| 번호 | 행동 |
| --- | --- |
| 0 | 아무 입력 없음 |
| 1 | 오른쪽 |
| 2 | 오른쪽 + A |
| 3 | 오른쪽 + B |
| 4 | 오른쪽 + A + B |

## 테스트

```bash
python -m unittest discover -s tests
```

테스트는 DQN 출력 모양, replay buffer batch, agent 행동 선택, Mario 전처리 wrapper, checkpoint 로딩, ZSNES 브릿지 유틸리티를 확인합니다.

## 참고

- GPU가 있으면 PyTorch가 CUDA를 사용합니다.
- CPU에서도 실행은 가능하지만 학습 속도는 느립니다.
- `checkpoints/`, `runs/`, `__pycache__/` 등 생성물은 저장소에 올리지 않는 것을 권장합니다.
