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

function run_hd() {
  input1=$1
  input2=$2
  input_type=$3
  n_dims=$4
  translate=$5

  name1=$(basename $input1)
  name2=$(basename $input2)

  log="${log_dir}/run_all/rt_analysis/${name1}_${name2}_translate_${translate}.json"

  echo "${log}" | xargs dirname | xargs mkdir -p

  if [[ -f "$log" ]]; then
    echo "Skipping, $log exists"
  else
    $PROG_ROOT/hd_exec \
      -input1 $input1 \
      -input2 $input2 \
      -input_type $input_type \
      -n_dims $n_dims \
      -serialize $SERIALIZE_ROOT \
      -variant "rt" \
      -execution "gpu" \
      -repeat 5 \
      -json "$log" \
      -check=false \
      -translate=$translate
  fi
}

function run_analysis_vary_dist() {
  for translate in 0.01 0.02 0.03 0.04 0.05 0.06. 0.07 0.08; do
    log="${log_dir}/run_all/rt_analysis/translate_${translate}.json"

    echo "${log}" | xargs dirname | xargs mkdir -p

    if [[ -f "$log" ]]; then
      echo "Skipping, $log exists"
    else
      $PROG_ROOT/hd_exec \
        -input1 "$DATASET_ROOT/geo/USADetailedWaterBodies.wkt" \
        -input2 "$DATASET_ROOT/geo/USACensusBlockGroupBoundaries.wkt" \
        -input_type "wkt" \
        -n_dims 2 \
        -serialize $SERIALIZE_ROOT \
        -variant "compare-methods" \
        -execution "gpu" \
        -repeat 1 \
        -json "$log" \
        -check=false \
        -rt_prune=false \
        -rt_eb=false \
        -translate=$translate
    fi
  done
}

function run_analysis_vary_max_hit() {
  for max_hit in 1 16 32 64 128 256 512 1024; do
    log="${log_dir}/run_all/rt_analysis/max_hit_${max_hit}.json"

    echo "${log}" | xargs dirname | xargs mkdir -p

    if [[ -f "$log" ]]; then
      echo "Skipping, $log exists"
    else
      $PROG_ROOT/hd_exec \
        -input1 "$DATASET_ROOT/geo/USADetailedWaterBodies.wkt" \
        -input2 "$DATASET_ROOT/geo/USACensusBlockGroupBoundaries.wkt" \
        -input_type "wkt" \
        -n_dims 2 \
        -serialize $SERIALIZE_ROOT \
        -variant "compare-methods" \
        -execution "gpu" \
        -repeat 1 \
        -json "$log" \
        -check=false \
        -rt_prune=false \
        -rt_eb=false \
        -translate=0.08 \
        -max_hit=$max_hit
    fi
  done
}

run_analysis_vary_dist
run_analysis_vary_max_hit
