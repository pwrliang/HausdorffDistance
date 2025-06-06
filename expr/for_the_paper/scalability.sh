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
  out_prefix=$1
  input1=$2
  input2=$3
  input_type=$4
  n_dims=$5
  variant=$6
  execution=$7
  limit=$8
  translate=$9

  name1=$(basename $input1)
  name2=$(basename $input2)

  log="${log_dir}/run_all/${variant}_${execution}/${out_prefix}/${name1}_${name2}_limit_${limit}_translate_${translate}.json"

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
      -variant $variant \
      -execution $execution \
      -repeat 1 \
      -json "$log" \
      -check=false \
      -translate=$translate \
      -limit $limit \
      -auto_tune
  fi
}

for size in 10000000 12000000 14000000 16000000 18000000 20000000; do
  dataset="$DATASET_ROOT/geo/all_nodes.wkt"
  for variant in eb rt hybrid; do
    run_hd "scal_vary_size" "$dataset" "$dataset" "wkt" 3 $variant "gpu" $size "0.005"
  done
done

for translate in 0.002 0.004 0.006 0.008 0.01; do
  dataset="$DATASET_ROOT/geo/all_nodes.wkt"
  for variant in eb rt hybrid; do
    run_hd "scal_vary_translate" "$dataset" "$dataset" "wkt" 3 $variant "gpu" 10000000 $translate
  done
done
