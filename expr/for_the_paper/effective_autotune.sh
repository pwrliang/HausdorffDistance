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
  normalize=$5
  n_dims=$6
  auto_n_points_cell=$7
  auto_max_hit=$8

  name1=$(basename $input1)
  name2=$(basename $input2)

  log="${log_dir}/run_all/auto_tune/${out_prefix}/n_points_cell_${auto_n_points_cell}_max_hit_${auto_max_hit}/${name1}_${name2}.json"

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
      -variant "hybrid" \
      -execution "gpu" \
      -repeat 5 \
      -json "$log" \
      -check=false \
      -normalize="$normalize" \
      -auto_tune_n_points_cell="$auto_n_points_cell" \
      -auto_tune_max_hit="$auto_max_hit"
  fi
}

function run_geo() {
  root="$DATASET_ROOT/geo"
  datasets1=(dtl_cnty.wkt USADetailedWaterBodies.wkt lakes.bz2.wkt)
  datasets2=(uszipcode.wkt USACensusBlockGroupBoundaries.wkt parks.bz2.wkt)
  for ((i = 0; i < ${#datasets1[@]}; i++)); do
    dataset1=${datasets1[i]}
    dataset2=${datasets2[i]}
    run_hd "geo" "$root/$dataset1" "$root/$dataset2" "wkt" "false" 2 "false" "false"
    run_hd "geo" "$root/$dataset1" "$root/$dataset2" "wkt" "false" 2 "true" "false"
    run_hd "geo" "$root/$dataset1" "$root/$dataset2" "wkt" "false" 2 "false" "true"
    run_hd "geo" "$root/$dataset1" "$root/$dataset2" "wkt" "false" 2 "true" "true"
  done
}

function run_graphics() {
  root="$DATASET_ROOT/graphics"
  datasets1=(dragon.ply thai_statuette.ply dragon.ply thai_statuette.ply)
  datasets2=(asian_dragon.ply happy_buddha.ply happy_buddha.ply asian_dragon.ply)
  for ((i = 0; i < ${#datasets1[@]}; i++)); do
    dataset1=${datasets1[i]}
    dataset2=${datasets2[i]}
    run_hd "graphics" "$root/$dataset1" "$root/$dataset2" "ply" "false" 3 "false" "false"
    run_hd "graphics" "$root/$dataset1" "$root/$dataset2" "ply" "false" 3 "true" "false"
    run_hd "graphics" "$root/$dataset1" "$root/$dataset2" "ply" "false" 3 "false" "true"
    run_hd "graphics" "$root/$dataset1" "$root/$dataset2" "ply" "false" 3 "true" "true"
  done
}

run_geo
run_graphics
