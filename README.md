# ESS 배터리 수명 예측

초기 100사이클의 측정값으로 LFP·흑연 배터리 **셀**의 총 사이클 수명을 예측했습니다. ESS의 교체 계획에 필요한 신호를 찾는 것이 목적입니다. 셀 실험값을 ESS 팩의 교체 시점으로 바로 해석하지는 않습니다.

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 데이터셋 | MIT–Stanford Battery Dataset (Severson et al., 2019) |
| 학습 | Batch 1 (2017-05-12), 수명 확인 46셀 |
| 평가 | Batch 2 (2018-02-20), 수명 확인 34셀 |
| 추가 평가 | Batch 3 (2018-04-12), 수명 확인 40셀 |
| 태스크 | 회귀: 초기 100사이클 → `cycle_life` 예측 |
| 수명 기준 | 방전 용량이 정격 용량의 80%에 도달하는 시점 |

원본에는 Batch 1·2·3 각각 46·47·46셀이 있습니다. 공개 코드의 배치 연속 측정 5셀과 노이즈 채널 6셀, 수명값이 없는 Batch 2의 8셀을 제외한 뒤 위 수명 분석 표본을 사용했습니다. 셀 하나를 한 표본으로 삼았습니다.

## 파일 구조

```text
Mini PJT/
├── data/README.md                  # 원본 파일 이름과 다운로드 위치
├── DAY1-EDA.ipynb                  # Day 1 EDA 확인
├── DAY1-REPORT.md                 # Day 1 해석
├── DAY2-REPORT.md                 # Day 2 성능·오류·개선 해석
├── src/
│   ├── day1_eda.py                # 배치별 셀/사이클 추출
│   ├── day1_compare.py            # 배치 비교와 그림
│   ├── day2_model.py              # 분할·CV·학습·평가
│   ├── day2_model_compare.py      # 20개 후보 비교
│   └── day2_batch_calibration.py  # 새 배치 레이블 보정 실험
├── results/
│   ├── model_performance.csv       # 제출 형식 성능표
│   └── day2/                      # 분할, 셀별 예측, 잔차 그림
└── requirements.txt
```

원본 `.mat` 파일은 Git에서 제외했습니다. 다운로드할 파일 목록은 [data/README.md](data/README.md)에 있습니다. 보고서 PDF는 [Day 1](DS-MINI-Design-울산_1반-박준형.pdf)과 [Day 2](DS-MINI-Design-울산_1반-박준형-DAY2.pdf)입니다.

## 환경 설정과 재현

Python 3.14에서 검증했습니다. 원본 파일을 `data/`에 놓은 뒤 프로젝트 루트에서 실행합니다. 큰 `.mat` 파일을 한 번에 메모리에 올리지 않도록 배치별 추출을 별도 프로세스로 실행합니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/day1_eda.py batch1
.venv/bin/python src/day1_eda.py batch2
.venv/bin/python src/day1_eda.py batch3
.venv/bin/python src/day1_compare.py
.venv/bin/python src/day2_model.py --stage select
.venv/bin/python src/day2_model.py --stage evaluate --model ridge_core_a1
.venv/bin/python src/day2_model_compare.py --stage screen
.venv/bin/python src/day2_model_compare.py --stage diagnose
.venv/bin/python src/day2_batch_calibration.py
```

PDF는 `.venv/bin/python src/build_day1_pdf.py`와 `.venv/bin/python src/build_day1_pdf.py --day 2`로 다시 만듭니다. 라이브러리 버전은 [requirements.txt](requirements.txt)에 고정했습니다.

## EDA

### 수명 분포

수명 중앙값은 Batch 1 **859**, Batch 2 **469**, Batch 3 **965**사이클입니다. Batch 2에는 500사이클 미만 셀이 **26/34셀**, Batch 3에는 1,000사이클 초과 셀이 **19/40셀**입니다. 이 데이터에서 Batch 1과 Batch 2가 비슷한 수명 분포라는 가정은 맞지 않았습니다. Batch 1만으로 학습한 모델이 Batch 2의 짧은 수명을 길게 예측할 위험을 예상했습니다.

### 방전 용량 열화와 knee

방전 용량 곡선을 두 구간 직선으로 근사하니 후반 감소가 더 가팔랐습니다. 탐색적 knee 시점 중앙값은 배치별 약 **602·351·766사이클**입니다. knee는 100사이클 이후 기록을 써야 계산되므로 예측 입력에서 제외했습니다.

### 초기 ΔQ(V)

같은 셀의 100사이클 `Qdlin(V)`에서 10사이클 값을 뺀 곡선의 분산을 구했습니다. `log10 var(ΔQ(V))`와 수명의 Spearman 상관은 Batch 1·2·3에서 **-0.87·-0.71·-0.76**이었습니다. 세 배치에서 방향이 유지돼 가장 중요한 변수로 골랐습니다. 곡선의 평균·최솟값은 같은 정보를 일부 공유하므로 대표 입력에서 제외했습니다.

### 충전 조건과 중복 변수

초기 충전 시간과 수명의 상관은 배치별 **0.61·-0.32·0.05**로 일정하지 않았습니다. 두 충전 구간을 반영한 정책 강도와 `ΔQ(V)` 로그 분산의 Batch 1 Pearson 상관은 **0.95**였습니다. 정책 변수를 추가했을 때 Batch 1 CV가 조금 좋아져도 독립적인 정보로 보지 않았습니다. `Tavg`와 `Tmax`도 강하게 겹쳐 평균 온도만 남겼습니다.

## Modeling

### 피처 엔지니어링 전략

최종 입력은 `log10 var(ΔQ(V))`, 초기 100사이클 평균 `QD`, 초기 100사이클 평균 `Tavg`입니다. 첫 변수는 초기 열화 신호, 나머지 둘은 시작 용량과 온도 조건을 함께 반영합니다. 수명 레이블과 미래 용량 기록은 입력하지 않았습니다.

### 파이프라인과 모델 선택

Batch 1의 46셀을 충전 정책별로 개발용 **35셀**과 hold-out **11셀**로 나눴습니다. 같은 정책이 양쪽에 겹치지 않습니다. 개발용에서는 정책별 5분할 `GroupKFold`를 사용했습니다. 결측 대체와 표준화는 각 학습 폴드에서만 `fit`했습니다. 무작위성은 `seed=42`로 고정했고 분할 목록은 `results/day2/batch1_split.csv`에 있습니다.

후보는 중앙값 기준선, 선형회귀·Ridge·Elastic Net·Huber, 로그 타깃 Ridge, SVR·KNN, 단일 트리, Random Forest·Extra Trees·Gradient Boosting의 **20개 설정**입니다. 최종 평가에는 **세 변수 Ridge(α=1)**를 사용했습니다. Day 1의 EDA와 개발용 CV에 근거해 hold-out·Batch 2 점수를 보기 전에 선택했습니다. 이후 확장 비교에서 로그 Ridge의 개발용 CV가 더 낮았지만 Batch 2 오차는 줄지 않았습니다. 트리 계열은 Batch 2 점수만 보고 사후에 교체하지 않았습니다.

## 성능 결과

| 구분 | MAPE (%) | 비고 |
|---|---:|---|
| Train (Batch 1 CV) | 6.66 | 개발용 35셀, 정책별 5분할 평균 |
| Valid (Batch 1 Hold-out) | 11.01 | 정책이 겹치지 않는 11셀 |
| Test (Batch 2) | 48.67 | Batch 1 전체 학습 후 34셀 평가 |
| Gap (Train-Valid) | +4.35%p | Valid − Train. 양수면 검증 오차 증가 |
| Gap (Valid-Test) | +37.66%p | Batch 2 − Valid. 양수면 배치 이동 시 오차 증가 |
| Gap (Target-Test) | +39.57%p | Batch 2 − 원논문 참고값 9.1% |
| Test (Batch 3) | 16.67 | 추가 평가 40셀 |
| Gap (Batch2-Batch3) | -32.00%p | Batch 3 − Batch 2. 음수면 Batch 3 오차가 낮음 |
| Gap (Target-Test, Batch 3) | +7.57%p | Batch 3 − 원논문 참고값 9.1% |

원자료 구성과 검증 방식이 원논문과 동일하지 않아 **9.1%는 직접 비교 가능한 재현 성능이 아니라 참고값**입니다. 정확한 수치는 [results/model_performance.csv](results/model_performance.csv), 후보별 비교는 `results/day2/model_comparison/comparison_table.csv`에 있습니다.

## 오류 분석

Batch 2의 실제 수명 중앙값은 **469사이클**, 예측 중앙값은 **741사이클**입니다. 34셀 중 32셀을 길게 예측했습니다. 오차가 큰 `batch2_029`는 실제 452사이클을 832사이클로, `batch2_006`은 393사이클을 731사이클로 예측했습니다. 가장 크게 틀린 셀은 주로 `newstructure` 표시가 없는 짧은 수명 셀이었습니다. 표시가 없는 28셀의 수명 중앙값은 **451사이클**, MAPE는 **55.5%**였고, 표시가 있는 6셀은 각각 **1,013사이클**, **16.8%**였습니다. 이 표시의 정확한 실험 의미와 원인 효과는 확인하지 못했습니다.

Batch 3에서는 1,000사이클 초과 19셀을 평균 **351사이클 짧게** 예측했습니다. `batch3_038`은 1,935사이클을 946사이클로 예측한 가장 큰 사례입니다. Batch 1의 최대 수명 1,227사이클을 넘어서는 구간이라 선형식의 외삽과 배치별 측정 조건 차이를 함께 봐야 합니다.

셀별 근거는 `results/day2/cell_predictions.csv`, 그룹별 요약은 `results/day2/error_group_summary.csv`, 오차 상위 셀은 `results/day2/worst_errors.csv`에 있습니다.

## ESS 도메인 해석과 한계

초기 곡선 변화가 큰 셀을 추가 관찰 대상으로 **선별하는 신호**로는 활용 가능성이 보였습니다. 그러나 Batch 2의 짧은 수명을 평균 249사이클 길게 예측한 모델을 ESS 교체 시점 결정에 바로 쓰면 늦은 교체 판단으로 이어질 수 있습니다. 실무 적용에는 셀 실험과 다른 팩 구성, 온도, 운전 프로토콜, BMS 측정값이 필요합니다.

새 배치에서 몇 개의 수명 레이블을 확보해 기준을 보정하는 방법은 가능성을 보였습니다. 정책이 겹치지 않게 Batch 2를 50번 나눈 탐색 실험에서 7~15개 보정 셀을 사용한 중앙 비율 보정의 평가 셀 평균 MAPE는 **15.72%**였습니다. 같은 셀이 여러 반복에 참여했고 이미 Batch 2 결과를 본 뒤 설계한 실험이므로 독립적인 최종 성능은 아닙니다. Batch 2 보정값을 Batch 3에 적용하면 오히려 악화됐습니다. 새 배치에서 레이블 확보·보정·평가 절차를 먼저 고정하고 별도 데이터로 검증해야 합니다.

## 참고문헌

- Severson et al. (2019), [*Data-driven prediction of battery cycle life before capacity degradation*](https://doi.org/10.1038/s41560-019-0356-8), *Nature Energy* 4, 383–391.
- [MIT–Stanford Battery Dataset](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)과 [원논문 공개 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation).

## 작성자

박준형(U015): 데이터 확인, EDA, 피처 설계, 모델 개발, Batch 2·3 평가와 보고서 작성.
