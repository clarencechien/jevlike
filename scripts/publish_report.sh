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
  -H "X-Title: Jev 式決策，值得，而且不用買 — 給長官的判定備忘" \
  --data-binary @results/report.html
echo
