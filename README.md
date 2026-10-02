# ESS 배터리 수명 예측

초기 100사이클의 측정값으로 LFP·흑연 배터리 **셀**의 총 사이클 수명을 예측했습니다. ESS의 점검·교체 계획에 참고할 초기 신호를 찾는 것이 목적입니다. 단일 셀 실험의 수명값을 ESS 팩의 교체 시점으로 바로 쓰지 않습니다.

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 데이터셋 | Severson 외(2019)의 MIT-Stanford 배터리 데이터 |
| 학습 | Batch 1 (`2017-05-12`), 수명 분석 41셀 |
| 필수 평가 | Batch 2 (`2017-06-30`), 수명 분석 43셀 |
| 추가 평가 | Batch 3 (`2018-04-12`), 수명 분석 40셀 |
| 태스크 | 회귀: 초기 100사이클 → `cycle_life` 예측 |
| 수명 기준 | 방전 용량이 정격의 80%에 도달하는 시점 |

원본은 46·48·46셀입니다. [원논문 공개 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/Load%20Data.ipynb)를 따라 Batch 1의 미종료 5셀을 제외하고, Batch 1 첫 5셀의 연속 측정 길이를 수명값에 반영했습니다. 이어 측정한 Batch 2의 5셀과 Batch 3의 노이즈 채널 6셀은 각 배치 평가에서 제외했습니다. 제외·보정 내역은 `results/all_cells.csv`와 `results/paper_screened_cells.csv`에 있습니다.

이 프로젝트의 Batch 2는 **`2017-06-30` 원본 파일**입니다. 분석에 사용한 파일 이름과 다운로드 경로는 [data/README.md](data/README.md)에 고정했습니다.

## 파일 구조

```text
Mini PJT/
├── data/README.md                  # 공식 원본 파일과 처리 기준
├── DAY1-EDA.ipynb                  # Day 1 EDA 확인
├── DAY1-REPORT.md                 # Day 1 분석과 전략
├── DAY2-REPORT.md                 # Day 2 평가와 해석
├── src/
│   ├── day1_eda.py                # 배치별 셀·사이클 추출
│   ├── day1_compare.py            # 배치 비교·그림·셀이력 보정
│   ├── day1_model_probe.py        # Day 1 사전 모델 점검
│   ├── report_extra_figures.py    # 배치 내 ΔQ(V)·knee 비교 그림
│   ├── day2_model.py              # 정책별 분할·CV·최종 평가
│   ├── day2_model_compare.py      # 21개 후보 비교
│   ├── day2_feature_ablation.py   # ΔQ·QD·Tavg 제거 실험
│   ├── day2_batch_calibration.py  # Batch 2 레이블 보정 실험
│   ├── day2_visuals.py            # 배치 이동·오차 시각화
│   ├── audit_results.py           # 원본·피처·분할·성능 재검증
│   └── build_day1_pdf.py          # 두 보고서의 PDF 생성
├── results/
│   ├── model_performance.csv       # 제출 형식 성능표
│   └── day2/                      # 분할·셀별 예측·후보 비교·그림
└── requirements.txt
```

원본 `.mat` 파일은 Git에 포함하지 않습니다. 다운로드할 파일 이름과 출처는 [data/README.md](data/README.md)에 있습니다. 보고서 PDF는 [Day 1](DS-MINI-Design-울산_1반-박준형.pdf)과 [Day 2](DS-MINI-Design-울산_1반-박준형-DAY2.pdf)입니다.

## 환경 설정과 재현

Python 3.14에서 확인했습니다. 세 원본 파일을 `data/`에 놓은 뒤 프로젝트 루트에서 실행합니다. 각 `.mat` 파일은 메모리 사용을 고려해 별도 프로세스로 추출합니다.

```bash
git clone https://github.com/jun394647/ess-battery-cycle-life-mini-project.git
cd ess-battery-cycle-life-mini-project
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/day1_eda.py batch1
.venv/bin/python src/day1_eda.py batch2
.venv/bin/python src/day1_eda.py batch3
.venv/bin/python src/day1_compare.py
.venv/bin/python src/report_extra_figures.py
.venv/bin/python src/day1_model_probe.py
.venv/bin/python src/day2_model_compare.py --stage screen
.venv/bin/python src/day2_feature_ablation.py
.venv/bin/python src/day2_model.py --stage select
.venv/bin/python src/day2_model.py --stage evaluate --model log_ridge_dq_qd
.venv/bin/python src/day2_model_compare.py --stage diagnose
.venv/bin/python src/day2_batch_calibration.py
.venv/bin/python src/day2_visuals.py
.venv/bin/python src/audit_results.py
.venv/bin/python src/build_day1_pdf.py
.venv/bin/python src/build_day1_pdf.py --day 2
```

## EDA와 피처 엔지니어링

Batch 1·2·3의 수명 중앙값은 **842·481·965사이클**입니다. Batch 2의 43셀 중 **33셀**이 500사이클 미만이고, Batch 3의 40셀 중 **19셀**이 1,000사이클 초과입니다. 두 구간 직선으로 근사한 방전 용량 곡선의 탐색적 knee 중앙값은 **580·381·766사이클**입니다. knee는 미래 기록이므로 입력에서 제외했습니다.

같은 셀의 100사이클과 10사이클 `Qdlin(V)` 차이 곡선에서 `log10 var(ΔQ(V))`를 만들었습니다. 수명과의 Spearman 상관은 **-0.88·-0.65·-0.76**으로 세 배치에서 방향이 유지됐습니다. 이 변수를 가장 중요하게 보고, 초기 평균 `QD`를 보조 입력으로 선택했습니다. `Tavg`와 충전 정책 강도는 배치별 관계·변수 중복·제거 실험을 근거로 최종 입력에서 제외했습니다. 수명 구간과 정책별 해석은 [Day 1 보고서](DAY1-REPORT.md)에 적었습니다.

## Modeling

Batch 1을 충전 정책별로 개발용 **31셀**과 hold-out **10셀**로 나눴고 같은 정책이 겹치지 않습니다. 개발용에서만 정책별 5분할 `GroupKFold`로 21개 설정을 비교했습니다. 결측 대체와 표준화는 각 학습 폴드에서만 맞췄습니다. 분할은 `results/day2/batch1_split.csv`에 있습니다.

**최종 모델은 `ΔQ(V)` 로그 분산과 초기 평균 `QD`를 쓰는 2변수 로그 타깃 Ridge**입니다. Batch 1 개발용 CV에서 7.59%로 3변수 모델의 8.07%보다 낮았고, `Tavg`의 배치별 수명 상관이 불안정했습니다. 이 모델은 하나의 Ridge 회귀이며 앙상블이 아닙니다. 후보별 사후 비교는 `results/day2/model_comparison/comparison_table.csv`에 있습니다.

2변수 제거 실험은 초기 3변수 모델의 Batch 2 결과를 본 뒤 추가했습니다. 직접 선택 기준은 Batch 1 개발용 CV였지만, 현재 Batch 2 점수는 **완전히 미접촉한 외부 검증**으로 해석하지 않습니다.

## 성능 결과

| 구분 | MAPE (%) | 비고 |
|---|---:|---|
| Train (Batch 1 CV) | **7.59** | 개발용 31셀, 정책별 5분할 평균 |
| Valid (Batch 1 Hold-out) | **7.82** | 정책이 겹치지 않는 10셀 |
| Test (Batch 2) | **24.11** | Batch 1 전체 학습 후 43셀 평가 |
| Gap (Train-Valid) | **+0.23%p** | Valid - Train |
| Gap (Valid-Test) | **+16.30%p** | Batch 2 - Valid |
| Gap (Target-Test) | **+15.01%p** | Batch 2 - 원논문 참고값 9.1% |
| Test (Batch 3) | **11.96** | 추가 평가 40셀 |
| Gap (Batch2-Batch3) | **-12.15%p** | Batch 3 - Batch 2 |
| Gap (Target-Test, Batch 3) | **+2.86%p** | Batch 3 - 9.1% |

9.1%는 원논문의 대표 테스트 오차입니다. 논문과 이 프로젝트의 모델·분할이 달라 재현 성패를 판정하는 동일 조건의 점수는 아닙니다. 정확한 수치는 [results/model_performance.csv](results/model_performance.csv)에 있습니다.

## 오류 분석

Batch 2의 실제 수명 중앙값은 **481사이클**, 예측 중앙값은 **596사이클**입니다. 43셀 중 37셀을 길게 예측했고, 500사이클 미만 33셀의 평균 편향은 **+103사이클**입니다. Batch 3에서는 1,000사이클 초과 19셀을 평균 **222사이클 짧게** 예측했습니다. 전체 MAPE와 함께 수명 구간별 오차 방향을 살펴야 한다고 판단했습니다.

그림은 [배치별 신호와 수명](results/day2/figures/feature_target_shift.png), [Batch 2 신호별 실제·예측](results/day2/figures/batch2_dq_calibration.png), [수명 구간별 잔차](results/day2/figures/error_by_life_band.png), [후보 모델의 배치 이동](results/day2/figures/candidate_transfer.png)에 있습니다. 셀별 예측은 `results/day2/cell_predictions.csv`, 오차가 큰 셀은 `results/day2/worst_errors.csv`에서 확인할 수 있습니다.

## ESS 도메인 해석과 한계

초기 곡선 변화는 같은 측정 조건에서 추가 관찰이 필요한 셀을 찾는 신호로 사용할 가능성이 있습니다. 하지만 짧은 수명을 평균 103사이클 길게 보는 모델을 ESS 교체 시점 결정에 바로 쓰면 늦은 대응으로 이어질 수 있습니다. 현장에는 팩의 온도·셀 편차·운전 이력·BMS 측정값이 더 필요합니다.

Batch 2의 일부 수명 레이블을 활용한 별도 보정 실험에서는 정책이 겹치지 않는 평가 셀의 평균 MAPE가 무보정 **24.02%**에서 중앙 비율 보정 **10.77%**로 줄었습니다. 매회 13~15개의 Batch 2 레이블을 사용한 **사후 실험**이므로 필수 테스트 성능으로 대체하지 않았습니다. Batch 2 전체 보정값을 Batch 3에 옮기면 MAPE가 **11.96→27.61%**로 악화됐습니다. 자세한 판단은 [Day 2 보고서](DAY2-REPORT.md)에 적었습니다.

## 평가 항목 점검

| 평가 항목 | 확인 위치 |
|---|---|
| Day 1 EDA: 분포·통계·피처 생성 | [Day 1 보고서의 EDA](DAY1-REPORT.md), `results/batch_summary.csv`, `results/feature_correlations.csv` |
| EDA에서 전략으로 이어지는 근거 | [Day 1 변수 선정과 모델 전략](DAY1-REPORT.md), `results/day1_probe_summary.csv` |
| 모델과 목표값 선택 논리 | [Day 1 모델 전략](DAY1-REPORT.md), [Day 2 후보 선정](DAY2-REPORT.md) |
| 전략의 코드 반영 | `src/day1_compare.py`, `src/day2_model.py`, `results/day2/protocol.json` |
| 분할·전처리·파이프라인 | `results/day2/batch1_split.csv`, `src/day2_model.py`, [원본 대조 결과](results/audit_results.json) |
| 성능 형식·논문 Gap·오류 분석 | [제출 형식 성능표](results/model_performance.csv), [Day 2 성능·오차 분석](DAY2-REPORT.md) |
| ESS 관점 해석과 한계 | [Day 2 ESS 도메인 해석과 제 생각](DAY2-REPORT.md) |

평가 항목에 필요한 분석과 설명을 모두 담았는지 위 위치에서 확인할 수 있습니다. `results/audit_results.json`은 원본 파일의 셀 수·수명 보정·초기 피처·정책 분할·셀별 예측·성능표를 원자료와 대조한 결과입니다. 목표 성능 9.1%는 달성하지 못했고, Batch 2의 배치 이동 오차를 한계로 명시했습니다.

## 참고문헌

- [Severson 외(2019), *Data-driven prediction of battery cycle life before capacity degradation*](https://doi.org/10.1038/s41560-019-0356-8), *Nature Energy* 4, 383-391.
- [MIT-Stanford 원자료](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)와 [공개 전처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation).

## 작성자

박준형(U015): 데이터 확인, EDA, 피처 설계, 모델 개발, Batch 2·3 평가와 보고서 작성.
