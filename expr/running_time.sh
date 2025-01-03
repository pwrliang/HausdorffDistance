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
  for file1 in "${DATASETS1[@]}"; do
    for file2 in "${DATASETS2[@]}"; do
      for dist in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
        log="${log_dir}/vary_dist/${file1}_${file2}_dist_${dist}_limit_$LIMIT.log"

        if [[ ! -f "${log}" ]]; then
          echo "${log}" | xargs dirname | xargs mkdir -p

          cmd="$PROG_ROOT/hd_exec \
          -input1 $DATASET_ROOT/$file1 \
          -input2 $DATASET_ROOT/$file2 \
          -serialize $SERIALIZE_ROOT \
          -limit $LIMIT \
          -move_offset 1"

          echo "$cmd" >"${log}.tmp"
          eval "$cmd" 2>&1 | tee -a "${log}.tmp"

          if grep -q "RT HausdorffDistance" "${log}.tmp"; then
            mv "${log}.tmp" "${log}"
          fi
        fi
      done
    done
  done
}

function cull() {
  LIMIT=1000000
  for cull in true false; do
    for file1 in "${DATASETS1[@]}"; do
      for file2 in "${DATASETS2[@]}"; do
        for dist in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
          log="${log_dir}/cull/${file1}_${file2}_dist_${dist}_limit_${LIMIT}_cull_${cull}.log"

          if [[ ! -f "${log}" ]]; then
            echo "${log}" | xargs dirname | xargs mkdir -p

            cmd="$PROG_ROOT/hd_exec \
                -input1 $DATASET_ROOT/$file1 \
                -input2 $DATASET_ROOT/$file2 \
                -serialize $SERIALIZE_ROOT \
                -limit $LIMIT \
                -move_offset $dist \
                -cull=$cull \
                -v=1"

            echo "$cmd" >"${log}.tmp"
            eval "$cmd" 2>&1 | tee -a "${log}.tmp"

            if grep -q "RT HausdorffDistance" "${log}.tmp"; then
              mv "${log}.tmp" "${log}"
            fi
          fi
        done
      done
    done
  done
}

#vary_dist
cull