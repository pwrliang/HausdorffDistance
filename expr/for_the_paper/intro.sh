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

  for translate in 0.01 0.02 0.04 0.08 0.16 0.32; do
    log="${log_dir}/intro/${variant}_${execution}_translate_${translate}.json"

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
        -parallelism 1 \
        -repeat 1 \
        -json "$log" \
        -overwrite
    fi
  done
}

vary_dist "eb" "cpu"
vary_dist "nn" "cpu"
