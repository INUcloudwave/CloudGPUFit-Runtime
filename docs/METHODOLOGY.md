# 예측 및 검증 방법

## 예측 구조

1. FX Graph 또는 `torch.export + FakeTensor`로 연산 종류, 실행 순서, Tensor shape와 dtype을 추출합니다.
2. 연산량과 메모리 접근량으로 Roofline 기준시간을 구합니다.
3. Linear·Attention·Pointwise·Reduction·Memory 등 연산군별 LightGBM으로 실제 GPU 실행 특성을 보정합니다.
4. Python 제어·디스패치·Framework Host 비용과 GPU 연산의 겹침을 순서 모델로 계산합니다.
5. H2D, Prefill, Decode 반복, D2H를 합산해 Request E2E 실행시간을 구합니다.

## Generate

```text
Generate = Prefill + Σ Decode_i
```

Decode는 KV cache 길이별 anchor graph의 예측값을 보간해 전체 출력 토큰 구간을 합산합니다. Decoder-only Eager에는 target별 CUDA-event 데이터로 학습한 Ridge 보정비를 anchor 시간에 적용합니다.

## 최종 개선 사항

- 하나의 공통 LightGBM에서 연산군별 LightGBM 구조로 분리
- CPU Host 제출과 GPU 실행의 겹침 반영
- Decoder-only Decode Ridge 보정
- `batch_norm`을 Reduction으로 분류
- `pad`, `constant_pad_nd`를 Memory로 분류
- 추론 상태 `dropout`을 실행시간이 없는 연산으로 처리

Vision 전용 Ridge는 모델 제외 검증에서 안정적인 개선이 없어 사용하지 않았습니다.

## 개선 효과 분리 검증

g6.xlarge의 실측 성공 446조건을 고정하고 예측 구조만 바꾸어 비교했습니다.

- 초기 통합형 LightGBM 43.99% → 연산군별 LightGBM 25.57%
- CPU와 GPU 시간을 단순 합산 59.77% → 실행 겹침 반영 15.95%

관련 행 단위 예측값과 집계 과정은 `results/experiments/g6.xlarge/e2e_ablation/`에 보존합니다.

## 검증 방식

- Eager: 인스턴스당 전체 470조건
- Compiled: 인스턴스당 모델별 대표 13조건
- 실측 성공 조건만 MAPE와 Bias 계산
- OOM과 실행 실패는 정확도 계산에서 제외하고 별도 집계

이 저장소의 결과는 저장 당시 검증 파이프라인의 최종 결과입니다. 제품 코드가 변경되면 `actual_results.jsonl`은 그대로 재사용하고 예측값만 다시 생성해 비교할 수 있습니다.
