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

source "${script_dir}/common.sh"

log_dir="${script_dir}/logs"

function vary_dist() {
  LIMIT=1000000
  variant=$1
  execution=$2

  for file1 in "${DATASETS1[@]}"; do
    for file2 in "${DATASETS2[@]}"; do
      for dist in 0.1 1; do # 0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8 25.6 51.2 102.4
        log="${log_dir}/vary_dist/${variant}_${execution}_${file1}_${file2}_dist_${dist}_limit_${LIMIT}.log"

        if [[ ! -f "${log}" ]]; then
          echo "${log}" | xargs dirname | xargs mkdir -p

          cmd="$PROG_ROOT/hd_exec \
                  -input1 $DATASET_ROOT/$file1 \
                  -input2 $DATASET_ROOT/$file2 \
                  -serialize $SERIALIZE_ROOT \
                  -limit $LIMIT \
                  -move_offset $dist \
                  -variant $variant \
                  -execution $execution \
                  -raymulticast 16"

          echo "$cmd" >"${log}.tmp"
          eval "$cmd" 2>&1 | tee -a "${log}.tmp"

          if grep -q "Running Time" "${log}.tmp"; then
            mv "${log}.tmp" "${log}"
          fi
        fi
      done
    done
  done
}

vary_dist "eb" "serial"
vary_dist "eb" "parallel"
vary_dist "eb" "gpu"
vary_dist "zorder" "serial"
vary_dist "rt" "gpu"

