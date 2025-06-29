#!/usr/bin/env bash

# Function to resolve the script path
get_script_dir() {
  local source="${BASH_SOURCE[0]}"
  while [ -h "$source" ]; do
    local dir
    dir=$(dirname "$source")
    source=$(readlink "$source")
    [[ $source != /* ]] && source="$dir/$source"
  done
  echo "$(cd -P "$(dirname "$source")" >/dev/null 2>&1 && pwd)"
}
script_dir=$(get_script_dir)

source "${script_dir}/../common.sh"

log_dir="${script_dir}/logs"

function vary_dist() {
  variant=$1
  execution=$2

  for translate in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10; do
    log="${log_dir}/analysis/${variant}_${execution}_translate_${translate}.json"

    if [[ ! -f "${log}" ]]; then
      echo "${log}" | xargs dirname | xargs mkdir -p

      $PROG_ROOT/hd_exec \
        -input1 "$DATASET_ROOT/ModelNet40/chair/train/chair_0653.off" \
        -input2 "$DATASET_ROOT/ModelNet40/chair/train/chair_0653.off" \
        -input_type off \
        -n_dims 3 \
        -serialize $SERIALIZE_ROOT \
        -translate $translate \
        -variant $variant \
        -execution $execution \
        -n_points_cell 8 \
        -repeat 1 \
        -json "$log" \
        -overwrite
    fi
  done
}


vary_dist "eb" "gpu"
#vary_dist "rt" "gpu"