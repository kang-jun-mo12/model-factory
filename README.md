# model-factory

[dino-flow-parking](https://github.com/kang-jun-mo12/dino-flow-parking)의 Flow 주차 플래너(StructureFlow)로 **하이퍼파라미터 검증용 모델을 여러 개 학습해서 모으는 저장소**입니다. 지금은 **스플라인 계수 K**를 4에서 24까지 바꿔 봅니다.

이 저장소에는 학습 코드와 학습된 Flow 모델만 둡니다. 학습 데이터(약 7 GB)와 앞단 가중치는 Release [`data-v1`](https://github.com/kang-jun-mo12/model-factory/releases/tag/data-v1)에 있습니다.

## 들어 있는 것

| 위치 | 내용 |
|---|---|
| `factory/` | 모델, 라벨, 데이터, 평가 코드. dino-flow-parking에서 학습에 필요한 부분만 옮겼고, K를 인자로 받게 고쳤습니다 |
| `scripts/download.py` | Release에서 데이터를 받아 `data/`에 넣고 SHA-256을 검사합니다 |
| `scripts/make_labels.py` | 전문가 궤적에서 계수 K의 라벨을 만듭니다 (`data/labels_K{K}.npz`) |
| `scripts/train.py` | 계수 K로 학습합니다. K 말고는 기준 모델과 같은 설정입니다 |
| `scripts/evaluate.py` | 검증 셋에서 경로 지표를 미터 단위로 잽니다 |
| `scripts/summarize.py` | `models/` 아래 모든 모델을 표 하나로 정리합니다 |
| `models/K12_drop40_s1/` | 기준 모델 (현재 최신 drop40_s1, K=12) |
| Release `data-v1` | `bev_0~3.npy`(Fused BEV), `trajectories.npz`(전문가 궤적), `ddp_frontend.pt`(앞단), `SHA256SUMS` |

## 학습 방법

Python 3.11과 CUDA GPU가 필요합니다. RTX 5070 Ti에서 1에폭이 약 50초, 100에폭이 약 1시간 30분입니다. BEV는 메모리 매핑으로 읽으니 RAM이 8 GB 이상 남아 있으면 빠릅니다.

**1. 설치와 데이터 받기**

```bash
git clone https://github.com/kang-jun-mo12/model-factory.git
```

```bash
cd model-factory
```

```bash
pip install -r requirements.txt
```

PyTorch는 자기 CUDA 버전에 맞는 것을 따로 설치하세요(시험한 버전: torch 2.11.0+cu128).

```bash
python scripts/download.py
```

`data/`에 약 7 GB가 받아집니다. 중간에 끊기면 다시 실행하면 이어 받습니다. 다른 디스크에 두려면 `--data 경로`를 쓰고, 아래 명령에도 같은 `--data`를 붙이세요.

**2. 계수 K의 라벨 만들기** (약 1분)

```bash
python scripts/make_labels.py -K 8
```

**3. 학습**

```bash
python scripts/train.py -K 8
```

`models/K08_drop40_s1/`에 `best.pt`, `train.log`, `meta.json`이 생깁니다. `meta.json`에는 설정, 검증 결과, 경로 지표, 코드 커밋, 데이터 해시, GPU가 들어갑니다.

**K 말고는 바꾸지 마세요.** 비교가 K 하나의 효과만 보도록 나머지는 기준 모델과 같게 둡니다: 시드 1, 100에폭, 배치 256, AdamW 3e-4와 OneCycle, scene-drop 0.4. 시드 반복을 맡은 경우에만 `--seed 2`, `--seed 3`을 씁니다.

## 맡을 실험

폴더 이름의 `K` 뒤 두 자리가 스플라인 계수입니다. 예를 들어 `K04_drop40_s1`은 계수 4, scene-drop 0.4, 시드 1입니다. 체크포인트 안의 `config.coefficients`와 `meta.json`의 `coefficients`에도 같은 값이 들어 있습니다.

| K (스플라인 계수) | 시드 1 | 시드 2 | 시드 3 |
|---|---|---|---|
| 4 | ✅ K04_drop40_s1 | | |
| 6 | ✅ K06_drop40_s1 | | |
| 8 | 학습 중 (kang-jun-mo12) | | |
| 10 | 학습 중 (kang-jun-mo12) | | |
| 12 | ✅ 기준 모델 | | |
| 14 | 학습 중 (kang-jun-mo12) | | |
| 16 | | | |
| 18 | | | |
| 20 | | | |
| 22 | | | |
| 24 | | | |

K는 4~24의 짝수입니다. 4는 3차 B-스플라인의 최소값입니다. 24는 짧은 구간 프레임 수(하위 5%가 12프레임)의 두 배이고, 그보다 크면 측정 잡음까지 맞추게 됩니다. 시작하기 전에 이슈를 열어 어떤 K와 시드를 맡는지 적어 주세요. 그래야 같은 실험을 두 번 하지 않습니다.

## 학습한 모델 올리기

`best.pt`는 약 11 MB라 git에 그대로 올립니다.

1. 이 저장소를 Fork합니다. 쓰기 권한이 있으면 Fork 없이 브랜치를 만들면 됩니다.
2. 새 브랜치에 `models/K{K}_drop40_s{시드}/` 폴더 하나만 추가합니다. 폴더 안에는 `best.pt`, `train.log`, `meta.json`가 있어야 하고, 학습이 만든 그대로 둡니다.
3. Pull Request를 엽니다. 템플릿의 확인 항목을 채워 주세요.

다른 파일(코드, 데이터)을 바꾸는 내용은 모델 PR과 섞지 말고 따로 올려 주세요. 코드가 바뀌면 이전 모델과 비교할 수 없게 될 수 있습니다.

## 결과 비교

```bash
python scripts/summarize.py
```

`models/` 아래 모든 모델의 `meta.json`을 표로 만듭니다. 다른 Euler 스텝으로 다시 재려면 다음처럼 실행합니다.

```bash
python scripts/evaluate.py models/K08_drop40_s1/best.pt --steps 4
```

**Flow 손실로 K를 비교하면 안 됩니다.** K가 바뀌면 잠재 벡터의 크기와 정규화가 달라지기 때문입니다. 그래서 `evaluate.py`는 경로를 디코드해서 미터 단위로 잽니다. 검증 창 1,024개에서 다음을 봅니다.

| 지표 | 뜻 |
|---|---|
| `end_gap_median`, `end_gap_p95` | 정답 구조로 샘플 1개를 뽑았을 때, 끝점이 라벨 경로 끝점에서 떨어진 거리 |
| `best_of_8_median`, `best_of_8_p95` | 플래너가 실제로 만드는 후보 8개(상위 구조 2개 × 4개) 중 라벨에 가장 가까운 끝점의 거리 |
| `mean_of_8_median` | 후보 8개 거리의 평균. 두 번째 구조의 후보가 들어가서 값이 큽니다 |
| `structure_hit` | 상위 2개 구조에 정답 구조가 있는 비율 |
| `over_limit` | 곡률이 조향 0.8 한계(0.345 1/m)를 한 번이라도 넘는 후보의 비율 |
| `ms_per_plan` | 후보 8개를 만드는 시간 (GPU마다 다르니 같은 GPU끼리만 비교) |
| `validation.correct` | 검증 셋 전체의 구조 정확도 |

이 지표는 1차로 거르는 용도입니다. 최종 판단은 CARLA 폐루프 주차 성공률로 합니다. 여기서 학습한 `best.pt`는 dino-flow-parking의 `plan_server.py`와 `benchmark.py`에서 그대로 읽힙니다. 계수 K는 체크포인트 설정에 저장되어 있습니다.

## 데이터

| 파일 | 내용 |
|---|---|
| `trajectories.npz` | 작업 866개, 205,757프레임의 전문가 궤적: 위치, 속도, 조향, 후진 여부, 목표 칸. 검증 작업 66개 표시 포함 |
| `bev_0~3.npy` | 학습 창 51,770개의 Fused BEV `[N,267,16,16]` float16. 2 GB 파일 한도 때문에 4개로 나눴습니다 |
| `ddp_frontend.pt` | BEV를 만든 고정된 DDP 앞단(DINOv2 ViT-B/14 + LSS). 학습에는 필요 없고 CARLA 평가에 필요합니다 (`download.py --frontend`) |

- **라벨**: 창은 4프레임(0.4초)마다 하나씩, 길이는 64프레임입니다. 라벨은 기준 모델과 같은 `squeeze` 방식(멈추는 깊이 -0.30 m)으로 만듭니다. `make_labels.py -K 12`는 기준 모델의 라벨을 비트 단위까지 똑같이 재현합니다.
- **분할**: 학습 47,807창(작업 800개), 검증 3,963창(작업 66개). 작업 단위로 나눠서, 이웃한 프레임이 학습과 검증에 걸치지 않습니다.
- **K는 경로 모양을 거의 바꾸지 않습니다.** 라벨 경로를 다시 그려 보면 끝점 오차가 K=4~32에서 모두 중앙값 0.057 m, 95% 0.112 m로 같습니다. K는 속도 곡선 적합(95% rms: K=4 0.23, K=12 0.16, K=24 0.095 m/s)과 학습 난이도에 영향을 줍니다.

## 출처와 사용 조건

이 저장소에는 아직 라이선스를 지정하지 않았습니다.

- **학습 데이터**: Hugging Face의 [`Jas000n/e2e_dataset`](https://huggingface.co/datasets/Jas000n/e2e_dataset)(작업 783개, 라이선스 표기 없음)과 [`ai4ce-drive/CarlaParking`](https://huggingface.co/datasets/ai4ce-drive/CarlaParking)의 `parkinglot_1000`(작업 83개, Apache-2.0, 접근 승인 필요)에서 나왔습니다. Release에는 원본 이미지가 없고, 앞단을 거친 특징과 궤적 수치만 있습니다.
- **앞단** `ddp_frontend.pt`: MIT로 표시된 Hugging Face 모델 [`ChampagneAndfragrance/Dino_Diffusion_Parking`](https://huggingface.co/ChampagneAndfragrance/Dino_Diffusion_Parking)의 `last.ckpt`에서 인식 모듈만 뽑은 것입니다. 학습은 원저자들이 했습니다. DINOv2 가중치가 포함되며 DINOv2는 Apache-2.0입니다.
- **코드**: dino-flow-parking에서 옮겼습니다. 주차 평가 프로토콜은 Dino-Diffusion Parking 공개 코드를 따릅니다.
