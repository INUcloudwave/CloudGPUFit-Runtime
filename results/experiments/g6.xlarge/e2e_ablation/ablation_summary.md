# g6.xlarge E2E ablation

- 기존 실측 비교 성공: **446개**
- LightGBM 세분화: **43.99% → 25.57%** (+18.42%p 개선)
- CPU·GPU 스케줄링: **59.77% → 15.95%** (+43.82%p 개선)

초기 통합형은 GEMM 계열 공용 모델과 Reduction·기타 계열 공용 모델을 사용한다.
세분화 모델은 Linear·Pointwise·Attention·Memory 등 실행 문맥별 전용 모델을 사용한다.
LightGBM 비교에서는 Decode Ridge와 Vision Ridge를 양쪽 모두 제외하였다.
CPU·GPU 비교에서는 연산별 LightGBM과 Host 모델을 동일하게 유지하고 합산 방식만 변경하였다.

## 전체 결과

| 방식 | MAPE(%) | 중앙 APE(%) | Bias(%) |
|---|---:|---:|---:|
| 초기 통합형 LightGBM | 43.99 | 43.23 | -43.99 |
| 연산군별 LightGBM 적용 | 25.57 | 23.50 | -19.05 |
| CPU+GPU 단순 합산 | 59.77 | 57.02 | 47.60 |
| CPU·GPU 오버랩 반영 | 15.95 | 9.38 | -7.37 |
