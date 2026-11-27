# 데이터셋 설명

## Target

| Target | GPU | VRAM |
|---|---|---:|
| g4dn.xlarge | NVIDIA T4 | 16GB |
| g5.xlarge | NVIDIA A10G | 24GB |
| g6.xlarge | NVIDIA L4 | 24GB |
| g6e.xlarge | NVIDIA L40S | 48GB |
| g7.2xlarge | NVIDIA RTX PRO 4500 Blackwell Server Edition | 32GB |
| g7e.2xlarge | NVIDIA RTX PRO 6000 Blackwell Server Edition | 96GB |

실제 드라이버·CUDA·Python·PyTorch·Transformers 버전은 각 target의 `environment/`에서 확인할 수 있습니다.

## Eager 데이터

- `actual_results.jsonl`: HF 모델을 실제 GPU에서 실행해 얻은 요청별 실측값
- `prediction_results_baseline.jsonl`: Decode Ridge와 최종 Vision 수정 전 예측값
- `prediction_results_ridge.jsonl`: Decoder-only Decode Ridge 적용 예측값
- `prediction_results_vision_fixed.jsonl`: g4dn·g5·g6의 Vision 연산 분류 수정 예측값
- `prediction_results_final.jsonl`: Ridge 결과에 Vision 수정 결과를 합친 최종 예측값
- `all_results.csv`: 실측·최종 예측·APE·Bias를 한 행에 결합한 표

g6e·g7·g7e는 Vision 연산 분류 수정이 측정 패키지에 이미 반영되어 있어 별도 `prediction_results_vision_fixed.jsonl`이 없습니다.

## Compiled Smoke 데이터

각 target에서 13개 모델의 대표 1조건을 `torch.compile` 환경으로 검증한 결과입니다.

- `actual_results.jsonl`
- `prediction_results.jsonl`
- `all_results.csv`

## Decode Ridge 데이터

`decode_ridge/decode_profile_rows.jsonl`은 single-Decode CUDA-event 프로필입니다. 주요 특징은 Batch, KV cache 길이, 예측 scheduled 시간, 연산군 비율, 모델 구조 정보입니다.

GPT-2·Qwen2.5·TinyLlama 데이터를 학습에 사용하고 OPT-125M을 holdout으로 확인한 뒤, 하나의 고정 Ridge(alpha=10) 보정기를 Decoder-only 6개 모델에 동일하게 적용했습니다.

## 개선 근거 실험 데이터

`results/experiments/g6.xlarge/`는 최종 정확도 데이터와 별도로 개선 방법의 효과를 확인하기 위한 자료입니다.

- `e2e_ablation/`: 동일한 446개 성공 조건에서 LightGBM 구조와 CPU·GPU 합산 방식을 비교한 예측값·집계표
- `cpu_gpu_trace/`: OPT-125M Prefill의 CPU·GPU 프로파일과 실행 구간 겹침 진단값
- `operator_profile/`: OPT-125M 내부 대표 연산의 Roofline, LightGBM 예측, CUDA Event 실측 비교

이 자료는 최종 2,820조건 정확도 집계에 추가 행으로 합산하지 않습니다. 같은 조건을 다른 예측 구조로 다시 계산한 ablation이기 때문입니다.

## 주요 실측 필드

| 필드 | 의미 |
|---|---|
| `case_id` | 모델·Batch·입력·출력 조건 식별자 |
| `status` | `ok`, `oom`, `failed` 등 실행 상태 |
| `actual_median_ms` | 반복 측정 요청시간 중앙값 |
| `actual_mean_ms` | 반복 측정 평균값 |
| `actual_std_ms` | 반복 측정 표준편차 |
| `actual_cv_percent` | 표준편차 ÷ 평균 × 100 |
| `samples_ms` | 개별 측정값 |
| `measurement_scope` | 실측 시작·종료 범위 |

## 오차 지표

```text
APE  = |예측값 - 실측값| / 실측값 × 100
MAPE = 비교 성공 조건의 APE 평균
Bias = (예측값 - 실측값) / 실측값 × 100의 평균
```

Bias가 음수이면 과소예측, 양수이면 과대예측입니다.
