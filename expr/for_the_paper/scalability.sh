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
  normalize=$8

  name1=$(basename $input1)
  name2=$(basename $input2)

  log="${log_dir}/run_all/${variant}_${execution}/${out_prefix}/${name1}_${name2}.json"

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
      -repeat 5 \
      -json "$log" \
      -check=false \
      -normalize="$normalize" \
      -translate=0.1 \
      -max_hit=1
  fi
}

for dist in uniform gaussian; do
  for size in 10000000 20000000 30000000 40000000 50000000 60000000; do
    dataset1="$DATASET_ROOT/synthetic/${dist}_seed_1_n_${size}.wkt"
    dataset2="$DATASET_ROOT/synthetic/${dist}_seed_2_n_${size}.wkt"
    for variant in eb rt hybrid; do
      run_hd "scalability" "$dataset1" "$dataset2" "wkt" 3 $variant "gpu" "false"
    done
  done
done