# ESS 배터리 수명 예측

초기 100사이클의 측정값으로 LFP·흑연 배터리 **셀**의 총 사이클 수명을 예측했습니다. 가장 중요하게 본 신호는 10사이클과 100사이클 사이의 `ΔQ(V)` 변화입니다. 배치가 바뀌어도 이 신호가 수명과 같은 방향으로 연결되는지 확인하고, 그 결과를 모델 선택에 반영했습니다.

**핵심 결과:** 초기 내부저항(`IR`)을 추가한 로그 Ridge의 Batch 1 정책별 5분할 CV MAPE는 **7.10%**, hold-out은 **7.15%**, 필수 평가인 Batch 2는 **19.39%**입니다. 이전 2변수 모델의 Batch 2 **24.11%**보다 낮아졌지만 배치 이동 오차는 여전히 큽니다. [Day 2 분석](DAY2-REPORT.md)에 모델을 바꾼 근거와 검증 한계를 적었습니다.

## 무엇을 예측했는가

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

## 저장소 구성

```text
Mini PJT/
├── assets/fonts/                   # PDF 한글 폰트와 라이선스
├── data/README.md                  # 공식 원본 파일과 처리 기준
├── DAY1-EDA.ipynb                  # 저장된 EDA 표·그림 열람
├── DAY1-REPORT.md                 # Day 1 분석과 전략
├── DAY2-REPORT.md                 # Day 2 평가와 해석
├── src/
│   ├── day1_eda.py                # 배치별 셀·사이클 추출
│   ├── day1_compare.py            # 배치 비교·그림·셀이력 보정
│   ├── day1_model_probe.py        # Day 1 사전 모델 점검
│   ├── report_extra_figures.py    # 배치 내 ΔQ(V)·knee 비교 그림
│   ├── day2_model.py              # 정책별 분할·CV·최종 평가
│   ├── day2_model_compare.py      # 22개 후보 비교
│   ├── day2_feature_ablation.py   # ΔQ·QD·Tavg 제거 실험
│   ├── day2_refinement.py         # 초기 내부저항 추가 실험
│   ├── day2_batch_calibration.py  # Batch 2 레이블 보정 실험
│   ├── day2_visuals.py            # 배치 이동·오차 시각화
│   ├── audit_results.py           # 원본·피처·분할·성능 재검증
│   ├── reproduce.py               # 전체 분석·보고서 순서대로 재생성
│   └── build_day1_pdf.py          # 두 보고서의 PDF 생성
├── results/
│   ├── model_inputs/               # 배치별 모델 입력; 개발 단계에서 Batch 1만 로드
│   ├── model_performance.csv       # 제출 형식 성능표
│   └── day2/                      # 분할·셀별 예측·후보 비교·그림
└── requirements.txt
```

원본 `.mat` 파일은 Git에 포함하지 않습니다. 다운로드할 파일 이름과 출처는 [data/README.md](data/README.md)에 있습니다. 보고서 PDF는 [Day 1](DS-MINI-Design-울산_1반-박준형.pdf)과 [Day 2](DS-MINI-Design-울산_1반-박준형-DAY2.pdf)입니다.

PDF의 한글 본문 폰트는 macOS에서 시스템 AppleGothic을 사용하고, 다른 환경에서는 저장소에 포함한 [NanumGothic Regular](https://github.com/google/fonts/tree/main/ofl/nanumgothic)을 사용합니다. 폰트 라이선스는 [OFL.txt](assets/fonts/OFL.txt)에 있습니다.

## 재현 방법

Python 3.14에서 확인했습니다. 아래 명령은 프로젝트 루트에서 실행합니다. 원본 `.mat` 세 파일의 합계는 약 **7.7GB**이므로 다운로드 전 디스크 공간을 확인해야 합니다. 파일 이름과 링크는 [data/README.md](data/README.md)에 있습니다.

```bash
git clone https://github.com/jun394647/ess-battery-cycle-life-mini-project.git
cd ess-battery-cycle-life-mini-project
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/reproduce.py
```

`reproduce.py`는 세 배치를 각각 별도 프로세스에서 추출한 다음 EDA, 후보 비교, 평가, 원자료 감사, PDF 생성 순서로 실행합니다. 원본 없이 Git에 포함된 셀 단위 결과에서 **모델과 PDF만** 재생성하려면 `.venv/bin/python src/reproduce.py --from-results`를 실행합니다. 이 모드는 원본 파일의 동일성이나 원본에서 피처를 추출한 과정을 검증하지 않습니다.

빠르게 결과를 확인할 때는 [Day 1 노트북](DAY1-EDA.ipynb)을 열면 됩니다. 노트북은 `results/`에 저장된 표와 그림을 보여주며 원자료를 재계산하지 않습니다. 저장된 결과와 실제 원본의 대조 기록은 [감사 결과](results/audit_results.json)에 있습니다. 원본이 없으면 `.venv/bin/python src/audit_results.py --artifacts-only`로 CV 재학습, 셀별 예측과 문서·성능표의 일치 여부를 확인할 수 있습니다.

## EDA에서 선택한 입력

Batch 1·2·3의 수명 중앙값은 **842·481·965사이클**입니다. Batch 2의 43셀 중 **33셀**이 500사이클 미만이고, Batch 3의 40셀 중 **19셀**이 1,000사이클 초과입니다. 두 구간 직선으로 근사한 방전 용량 곡선의 탐색적 knee 중앙값은 **580·381·766사이클**입니다. knee는 미래 기록이므로 입력에서 제외했습니다.

같은 셀의 100사이클과 10사이클 `Qdlin(V)` 차이 곡선에서 `log10 var(ΔQ(V))`를 만들었습니다. 수명과의 Spearman 상관은 **-0.88·-0.65·-0.76**으로 세 배치에서 방향이 유지됐습니다. 이 변수를 가장 중요하게 보고, 초기 평균 `QD`를 보조 입력으로 선택했습니다. `Tavg`와 충전 정책 강도는 배치별 관계·변수 중복·제거 실험을 근거로 최종 입력에서 제외했습니다. 수명 구간과 정책별 해석은 [Day 1 보고서](DAY1-REPORT.md)에 적었습니다.

## 모델 개발과 검증

Batch 1을 충전 정책별로 개발용 **31셀**과 hold-out **10셀**로 나눴고 같은 정책이 겹치지 않습니다. 개발용에서만 정책별 5분할 `GroupKFold`로 **22개 후보**를 비교했습니다. 후보 선택·변수 제거 단계는 `results/model_inputs/batch1.csv`만 열며, Batch 2·3 파일은 평가·사후 진단 단계에서 엽니다. 결측 대체와 표준화는 각 학습 폴드에서만 맞췄습니다. 분할은 `results/day2/batch1_split.csv`에 있습니다.

Day 1 사전 비교에는 Batch 1 전체 41셀이 사용됐습니다. 따라서 Day 2 hold-out은 **정책 분리 검증**이지만 처음부터 미사용한 셀은 아닙니다. Batch 2 역시 Day 1 EDA와 이전 2변수 모델 평가에서 노출됐습니다. 이 점을 점수의 해석 범위에 포함했습니다.

**현재 제출 모델은 `ΔQ(V)` 로그 분산, 초기 평균 `QD`, 초기 평균 내부저항 `IR`을 쓰는 로그 타깃 Ridge(α=0.1)**입니다. 이전 2변수 모델보다 Batch 1 개발용 CV가 **7.59→7.10%**로 낮았습니다. 5개 폴드 중 3개에서 개선됐고, 정책별 분할을 4·5·6개로 바꾸어도 개선 방향이 유지됐습니다. `IR`의 단순 수명 상관은 약하지만 `ΔQ(V)`와 완전히 겹치지 않는 초기 측정값으로 보았습니다. 여러 조합을 탐색한 뒤 고른 결과라 이 작은 CV 차이를 확정적인 우위로 보지 않습니다. 단일 Ridge 회귀이며 앙상블은 아닙니다. 세부 비교는 [추가 변수 실험](results/day2/refinement_summary.csv)과 [후보 비교](results/day2/model_comparison/comparison_table.csv)에 있습니다.

이번 `IR` 추가도 기존 모델의 Batch 2 결과를 본 뒤 진행했습니다. 따라서 Batch 2에서 **24.11→19.39%**로 낮아진 수치는 **사후 개선 결과**입니다. 완전히 미접촉한 외부 검증 성능으로 해석하지 않습니다. 선택 이력과 다음 검증 절차는 [Day 2 보고서](DAY2-REPORT.md)에 기록했습니다.

## 검증 성능

| 구분 | MAPE (%) | 비고 |
|---|---:|---|
| Train (Batch 1 CV) | **7.10** | 개발용 31셀, 정책별 5분할 평균 |
| Valid (Batch 1 Hold-out) | **7.15** | 정책이 겹치지 않는 10셀 |
| Test (Batch 2) | **19.39** | Batch 1 전체 학습 후 43셀 평가 |
| Gap (Train-Valid) | **+0.06%p** | Valid - Train |
| Gap (Valid-Test) | **+12.24%p** | Batch 2 - Valid |
| Gap (Target-Test) | **+10.29%p** | Batch 2 - 원논문 참고값 9.1% |
| Test (Batch 3) | **10.99** | 추가 평가 40셀 |
| Gap (Batch2-Batch3) | **-8.40%p** | Batch 3 - Batch 2 |
| Gap (Target-Test, Batch 3) | **+1.89%p** | Batch 3 - 9.1% |

9.1%는 과제에서 비교 기준으로 제시한 **논문 초록의 대표 수치**입니다. 논문 표의 Primary test 전체 셀 MAPE는 Full 모델 **14.1%**, Discharge 모델 **13.0%**입니다. 논문은 Batch 1·2를 섞어 학습·시험으로 나눴고, 이 프로젝트는 Batch 1만 학습해 Batch 2 전체를 시험했습니다. 따라서 9.1%를 논문의 동일한 Batch 2 전체 셀 점수로 해석하지 않았습니다. 자세한 근거는 [추가 실험과 논문 비교](DAY2-REPORT.md)에, 제출 형식 수치는 [성능표](results/model_performance.csv)에 있습니다.

## 9.1% 목표를 다시 시험한 결과

Batch 1에서만 후보를 비교하고 이후 Batch 2를 사후 평가했습니다. 처음 100사이클의 용량·저항 변화를 추가하고, 논문의 방전 곡선 변수도 구현했습니다. **내부 CV가 낮아진 후보도 Batch 2에서는 현재 모델보다 오차가 컸습니다.**

| 모델 또는 변경 | Batch 1 CV | Batch 1 hold-out | Batch 2 |
|---|---:|---:|---:|
| 현재 3변수 로그 Ridge | **7.10%** | **7.15%** | **19.39%** |
| 초기 평균 `QD`를 2번째 사이클 `QD`로 교체 | 6.46% | 8.23% | 23.58% |
| 논문형 방전 변수 6개, 이상 용량값 제외 | 5.75% | 10.46% | 23.11% |
| 짧은 수명 셀에 학습 가중치 추가 | 6.97% | 7.34% | 19.81% |
| 새 배치의 정답 없이 입력 분포로 가중치 조정 | 6.94% | 6.77% | 19.91% |

논문의 공개 분할처럼 Batch 1·2를 섞어 현재 모델을 다시 학습·시험하면 MAPE는 **11.16%**였습니다. 이는 분할의 영향을 살펴본 사후 진단이며 과제 필수 성능을 대체하지 않습니다. 실험별 변수 정의, 측정값 이상치, 모델 유지 판단은 [Day 2 보고서](DAY2-REPORT.md)에 적었습니다. 추가 실험 코드는 `src/day2_trajectory_experiment.py`, `src/day2_paper_feature_experiment.py`, `src/day2_shortlife_experiment.py`, `src/day2_covariate_shift_experiment.py`, `src/day2_gap_alignment_experiment.py`, `src/day2_paper_split_diagnostic.py`에 있습니다. 원자료가 필요한 실험도 있으므로 기본 `reproduce.py --from-results`에는 포함하지 않았습니다.

## 배치별 오차

Batch 2의 실제 수명 중앙값은 **481사이클**, 예측 중앙값은 **570사이클**입니다. 43셀 중 37셀을 길게 예측했고, 500사이클 미만 33셀의 평균 편향은 **+78사이클**입니다. Batch 3에서는 1,000사이클 초과 19셀을 평균 **177사이클 짧게** 예측했습니다. 이전 모델보다 오차는 줄었지만 방향은 같으므로 수명 구간별 편향을 함께 봅니다.

그림은 [배치별 신호와 수명](results/day2/figures/feature_target_shift.png), [Batch 2 신호별 실제·예측](results/day2/figures/batch2_dq_calibration.png), [수명 구간별 잔차](results/day2/figures/error_by_life_band.png), [후보 모델의 배치 이동](results/day2/figures/candidate_transfer.png)에 있습니다. 셀별 예측은 `results/day2/cell_predictions.csv`, 오차가 큰 셀은 `results/day2/worst_errors.csv`에서 확인할 수 있습니다.

## ESS 운영에서의 해석

초기 곡선 변화는 같은 측정 조건에서 추가 관찰이 필요한 셀을 찾는 신호로 사용할 가능성이 있습니다. 하지만 짧은 수명을 평균 78사이클 길게 보는 모델을 ESS 교체 시점 결정에 바로 쓰면 늦은 대응으로 이어질 수 있습니다. 현장에는 팩의 온도·셀 편차·운전 이력·BMS 측정값이 더 필요합니다.

Batch 2의 일부 수명 레이블을 활용한 별도 보정 실험에서는 정책이 겹치지 않는 평가 셀의 평균 MAPE가 무보정 **19.25%**에서 중앙 비율 보정 **10.90%**로 줄었습니다. 매회 13~15개의 Batch 2 레이블을 사용한 **사후 실험**이므로 필수 테스트 성능으로 대체하지 않았습니다. Batch 2 전체 보정값을 Batch 3에 옮기면 MAPE가 **10.99→20.65%**로 악화됐습니다. 자세한 판단은 [Day 2 보고서](DAY2-REPORT.md)에 적었습니다.

## 평가 기준과 산출물

| 평가 항목 | 확인 위치 |
|---|---|
| Day 1 EDA: 분포·통계·피처 생성 | [Day 1 보고서의 EDA](DAY1-REPORT.md), `results/batch_summary.csv`, `results/feature_correlations.csv` |
| EDA에서 전략으로 이어지는 근거 | [Day 1 변수 선정과 모델 전략](DAY1-REPORT.md), `results/day1_probe_summary.csv` |
| 모델과 목표값 선택 논리 | [Day 1 모델 전략](DAY1-REPORT.md), [Day 2 후보 선정](DAY2-REPORT.md) |
| 전략의 코드 반영 | `src/day1_compare.py`, `src/day2_model.py`, `results/day2/protocol.json` |
| 분할·전처리·파이프라인 | `results/day2/batch1_split.csv`, `src/day2_model.py`, [원본 대조 결과](results/audit_results.json) |
| 성능 형식·논문 Gap·오류 분석 | [제출 형식 성능표](results/model_performance.csv), [Day 2 성능·오차 분석](DAY2-REPORT.md) |
| ESS 관점 해석과 한계 | [Day 2 ESS 도메인 해석과 제 생각](DAY2-REPORT.md) |

평가 항목에 필요한 분석과 설명을 모두 담았는지 위 위치에서 확인할 수 있습니다. `results/audit_results.json`은 원본 파일의 셀 수·수명 보정·초기 피처·정책 분할·셀별 예측·성능표를 원자료와 대조한 결과입니다. 과제의 참고 성능 9.1%는 달성하지 못했고, Batch 2의 배치 이동 오차와 사후 모델 수정의 한계를 명시했습니다.

## 자료 출처

- [Severson 외(2019), *Data-driven prediction of battery cycle life before capacity degradation*](https://doi.org/10.1038/s41560-019-0356-8), *Nature Energy* 4, 383-391.
- [MIT-Stanford 원자료](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)와 [공개 전처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation).

## 작성

박준형(U015): 데이터 확인, EDA, 피처 설계, 모델 개발, Batch 2·3 평가와 보고서 작성.
