# CloudGPUFit Runtime — 추론 실행시간 예측 실측 데이터

CloudGPUFit 졸업작품의 **추론 실행시간 예측 파트**에서 수집한 AWS GPU 실측값, 예측값, Decode 보정 데이터와 최종 분석 결과를 정리한 저장소입니다.

웹·추천 API·예측기 제품 코드는 포함하지 않습니다. 제품 구현은 `CloudGPUFit` 저장소의 `orchestrator/eager_runtime/`에서 관리합니다.

## 한 줄 결론

- FP16 단일 GPU·단일 요청의 Request E2E 실행시간을 예측했습니다.
- 기본 실행 모드 전체 2,820조건 중 **2,704조건 비교 성공**, OOM 116조건이며, 성공 조건 통합 MAPE는 **15.23%**입니다.
- 최적화 적용 모드는 모델별 대표 1조건의 Smoke 검증이며, 78조건 통합 MAPE는 **17.55%**입니다.
- Decoder-only Decode에는 CUDA-event 데이터로 학습한 Ridge 보정을 적용했습니다.
- 이미지 분류는 누락 연산 분류를 수정했지만 GPU별 cuDNN 커널 선택 차이로 일부 오차가 남았습니다.
- g6.xlarge에서 연산군별 LightGBM, CPU·GPU 실행 겹침, 대표 연산 실측의 개선 근거를 별도 실험으로 검증했습니다.

## 예측 및 측정 범위

Request E2E는 CPU에 입력 Tensor가 준비된 시점부터 GPU 연산 결과가 CPU로 복사되어 반환되는 시점까지입니다.

포함 범위:

```text
입력 Tensor의 GPU 전송 → Prefill 또는 단일 Forward → Decode 반복 → 결과 Tensor의 CPU 전송
```

제외 범위:

- 모델 다운로드와 가중치 로딩
- 토큰화와 역토큰화
- 사용자와 서버 사이의 네트워크 지연
- 최적화 적용 모드의 최초 컴파일
- 다중 요청 배칭 및 다중 GPU 병렬 처리

## 최종 검증 결과

| 인스턴스 | Eager 성공/OOM | Eager MAPE | Compiled Smoke MAPE |
|---|---:|---:|---:|
| g4dn.xlarge | 427 / 43 | 15.13% | 17.48% |
| g5.xlarge | 446 / 24 | 13.73% | 14.97% |
| g6.xlarge | 446 / 24 | 13.22% | 16.52% |
| g6e.xlarge | 461 / 9 | 14.79% | 16.11% |
| g7.2xlarge | 454 / 16 | 17.11% | 21.11% |
| g7e.2xlarge | 470 / 0 | 17.24% | 19.12% |
| **전체** | **2,704 / 116** | **15.23%** | **17.55%** |

Eager는 모델·입력 조건 전체 470조건을 인스턴스마다 검증했습니다. Compiled는 각 모델의 대표 1조건, 인스턴스당 13조건을 검증했습니다.

## 폴더 구조

```text
CloudGPUFit-Runtime/
├── README.md
├── docs/
│   ├── DATASET.md
│   └── METHODOLOGY.md
├── analysis/
│   └── summarize.py
└── results/
    ├── summary/
    │   ├── eager_by_target.csv
    │   ├── eager_by_target_workload.csv
    │   ├── eager_by_target_model.csv
    │   ├── compiled_smoke_by_target.csv
    │   └── manifest.json
    ├── targets/<인스턴스>/
    │   ├── eager/
    │   │   ├── actual_results.jsonl
    │   │   ├── prediction_results_baseline.jsonl
    │   │   ├── prediction_results_ridge.jsonl
    │   │   ├── prediction_results_final.jsonl
    │   │   └── all_results.csv
    │   ├── compiled_smoke/
    │   ├── decode_ridge/
    │   └── environment/
    ├── experiments/g6.xlarge/
    │   ├── e2e_ablation/
    │   ├── cpu_gpu_trace/
    │   └── operator_profile/
    └── examples/g6-opt125m/
```

## 개선 근거 실험

`results/experiments/g6.xlarge/`에는 발표와 보고서에 사용한 다음 원본 자료가 포함됩니다.

| 실험 | 비교 범위 | 결과 |
|---|---|---|
| LightGBM 구조 변경 | 동일한 446개 성공 조건 | E2E MAPE 43.99% → 25.57% |
| CPU·GPU 실행 겹침 | 동일한 446개 성공 조건 | E2E MAPE 59.77% → 15.95% |
| OPT-125M 대표 연산 | Linear·LayerNorm·ReLU·Attention BMM | Roofline·보정 예측·CUDA Event 실측 비교 |

최종 정확도 표는 모든 개선을 반영한 결과이고, 위 수치는 각 개선 항목의 효과를 분리해 확인한 ablation 결과입니다.

## 모델과 입력 조건

| 워크로드 | 모델 | 조건 |
|---|---|---|
| Decoder-only | OPT-125M, TinyLlama-1.1B, Qwen2.5-1.5B, GPT-2, GPT-2 Medium, GPT-2 XL | Batch 1·8·16·32·128, 입력·출력 길이 128·256·512 |
| Encoder–Decoder | T5-Small, T5-Base, T5-3B | Batch 1·8·16·32·128, 입력·생성 길이 128·256·512 |
| 이미지 분류 | ResNet-18, ResNet-50, MobileNetV2 | Batch 1·8·16·32·128, 해상도 128·224·256·512 |
| 이미지 분류 | ViT-Large | Batch 1·8·16·32·128, 해상도 224 |

총 13개 모델, 인스턴스당 470조건입니다.

## 결과 다시 집계하기

추가 패키지 없이 Python 표준 라이브러리만 사용합니다.

```bash
python analysis/summarize.py --results results
```

이 명령은 인스턴스별 Ridge 예측에 Vision 수정 결과를 반영해 `prediction_results_final.jsonl`과 `all_results.csv`를 다시 만들고, `results/summary/`의 집계표를 갱신합니다.

## 주의

- P90은 Generate 전체 범위에서 보정·coverage 검증이 완료되지 않아 진단값으로만 사용합니다.
- OOM 조건은 MAPE에서 제외하고 별도로 집계합니다.
- 기본 실행 모드의 최종 Decoder-only 결과에는 Ridge 보정이 적용되어 있습니다.
- Compiled 결과에는 Decode Ridge가 적용되지 않습니다.
- PyTorch Eager와 `torch.compile` 기반 최적화만 다루며 vLLM·SGLang 같은 별도 서빙 엔진은 포함하지 않습니다.
