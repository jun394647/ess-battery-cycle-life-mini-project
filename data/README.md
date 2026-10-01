# 원본 데이터 배치

원본 `.mat` 파일은 용량 때문에 저장소에 포함하지 않습니다. [MIT–Stanford 배터리 데이터셋](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)에서 다음 파일을 내려받아 이 폴더에 놓습니다.

- `2017-05-12_batchdata_updated_struct_errorcorrect.mat` — Batch 1, 학습
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat` — Batch 2, 평가
- `2018-04-12_batchdata_updated_struct_errorcorrect.mat` — Batch 3, 추가 평가

`2018-04-03` varcharge 파일은 이 분석에 사용하지 않습니다. 이 프로젝트는 Kaggle 데이터셋 v1을 기준으로 실행했습니다. 셀별 가공 데이터와 평가 결과는 `results/`에 있습니다.
