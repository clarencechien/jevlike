#!/usr/bin/env bash
# Re-render results/report.html and publish it to imitator as public.
# Needs IMITATOR_TOKEN in the environment (never commit it) and a local copy of report.css.
set -euo pipefail
cd "$(dirname "$0")/.."
CSS="${1:-/tmp/report.css}"
[ -f "$CSS" ] || curl -sSL -o "$CSS" https://raw.githubusercontent.com/clarencechien/imitator/main/style/report.css
python3 bench/render_html_report.py "$CSS"
curl -sS -X PUT "https://imitator.ai-apps.work/v1/a/gb10-typed-decisions" \
  -H "Authorization: Bearer $IMITATOR_TOKEN" \
  -H "Content-Type: text/html" \
  -H "X-Visibility: public" \
  -H "X-Title: 八個問題，答了五個，三個給了上界 — GB10 typed decision 實驗紀錄" \
  --data-binary @results/report.html
echo
