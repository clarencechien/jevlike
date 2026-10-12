#!/usr/bin/env bash
# Re-render results/tw-report.html (the tw branch's script-dependence one-pager) and publish it to imitator as public.
# Needs IMITATOR_TOKEN in the environment (never commit it) and a local copy of report.css.
set -euo pipefail
cd "$(dirname "$0")/.."
CSS="${1:-/tmp/report.css}"
[ -f "$CSS" ] || curl -sSL -o "$CSS" https://raw.githubusercontent.com/clarencechien/imitator/main/style/report.css
python3 bench/render_tw_report.py "$CSS"
curl -sS -X PUT "https://imitator.ai-apps.work/v1/a/jevlike-tw-script" \
  -H "Authorization: Bearer $IMITATOR_TOKEN" \
  -H "Content-Type: text/html" \
  -H "X-Visibility: public" \
  -H "X-Title: 同一句話，換成簡體，答案就翻面 — Gemma 4 決策 API 的字形對照報告" \
  --data-binary @results/tw-report.html
echo
