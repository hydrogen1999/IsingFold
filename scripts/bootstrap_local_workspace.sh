#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "${script_dir}/.." && pwd)"
venv_root="${workspace_root}/.venv"
python_bin="${venv_root}/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  bootstrap_python="${PYTHON:-python3}"
  "${bootstrap_python}" -m venv "${venv_root}"
fi

"${python_bin}" -m pip install -e "${workspace_root}/packages/EmbedBench[dev,models]"
"${python_bin}" -m pip install -e "${workspace_root}[dev,baseline,figures,rl]"
