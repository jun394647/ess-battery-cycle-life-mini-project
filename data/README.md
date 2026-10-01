# 원본 데이터 배치

원논문에서 사용한 세 배치를 [MIT–Stanford 원자료 저장소](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)에서 내려받아 이 폴더에 놓습니다. 원본 `.mat` 파일은 용량 때문에 Git에 포함하지 않습니다.

- [Batch 1 다운로드](https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download) — `2017-05-12_batchdata_updated_struct_errorcorrect.mat`, 원본 46셀
- [Batch 2 다운로드](https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download) — `2017-06-30_batchdata_updated_struct_errorcorrect.mat`, 원본 48셀
- [Batch 3 다운로드](https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download) — `2018-04-12_batchdata_updated_struct_errorcorrect.mat`, 원본 46셀

원자료의 [공개 전처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/Load%20Data.ipynb)에 따라 Batch 1의 미종료 5셀을 제외하고, Batch 1 첫 5셀의 Batch 2 연속 측정 길이를 수명값에 더합니다. 연속 측정 5셀은 Batch 2에서 제외합니다. Batch 3의 노이즈 채널 6셀도 제외합니다. 분석 표본은 **41·43·40셀**입니다.

`2018-02-20` 파일은 다른 실험 자료입니다. 이전 분석에서 이 파일을 Batch 2로 잘못 사용한 오류를 정정했습니다. 현재 코드와 보고서는 `2017-06-30` 파일을 사용합니다. 셀별 처리 결과는 `results/all_cells.csv`와 `results/paper_screened_cells.csv`에서 확인할 수 있습니다.
