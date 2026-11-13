#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/postprocess_attentivefp_classification_outputs.sh"
bash "${SCRIPT_DIR}/postprocess_attentivefp_regression_outputs.sh"
