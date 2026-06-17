# Seoul Subway Ridership Data

## Source

- Dataset: 서울교통공사_역별 일별 시간대별 승하차인원 정보
- Provider: 서울교통공사
- Source page: https://data.seoul.go.kr/dataList/OA-12921/F/1/datasetView.do
- Original file: `서울교통공사_역별 일별 시간대별 승하차인원_20251231.csv`
- Downloaded: 2026-06-12
- Coverage: 2025-01-01 through 2025-12-31
- License: 공공누리 제1유형(출처표시)

## Local Extract

`seoul_subway_daily_2025.csv.gz` aggregates the original boarding and
alighting rows into one row per date, line, and station.

- Rows: 99,645
- Station-line combinations: 273
- Time bands: before 06:00 through after 24:00
- Additional fields: boarding total, alighting total, morning rush,
  evening rush, and total ridership
- Encoding: UTF-8 with BOM

The compressed extract is stored locally so the notebook can be rerun
without an API key or another 24 MB download.
