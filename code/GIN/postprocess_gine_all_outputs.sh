#!/usr/bin/env bash
set -euo pipefail
bash "$(dirname "${BASH_SOURCE[0]}")"/postprocess_gine_classification_outputs.sh
bash "$(dirname "${BASH_SOURCE[0]}")"/postprocess_gine_regression_outputs.sh
