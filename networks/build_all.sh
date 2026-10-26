#!/usr/bin/env bash
# Rebuild and validate every scenario from configs/scenarios/ (run from the repo root).
set -euo pipefail
python -m trafficmarl scenarios build all
python -m trafficmarl scenarios validate all
