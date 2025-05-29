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
  for translate in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08; do
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
  for max_hit in 1 2 4 8 16 32 64 128 256 512 1024; do
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
        -translate=0.05 \
        -max_hit=$max_hit
    fi
  done
}

function run_analysis_hit_count() {
  dataset_labels=("MRI" "CAD" "Geospatial" "Graphics")
  dataset_types=("image" "off" "wkt" "ply")
  dataset_dims=(3 3 2 3)
  norm_flags=("false" "true" "false" "false")
  datasets1=("$DATASET_ROOT/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_011/BraTS20_Validation_011_flair.nii" "$DATASET_ROOT/ModelNet40/curtain/train/curtain_0001.off" "$DATASET_ROOT/geo/USADetailedWaterBodies.wkt" "$DATASET_ROOT/graphics/dragon.ply")
  datasets2=("$DATASET_ROOT/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_092/BraTS20_Validation_092_t1ce.nii" "$DATASET_ROOT/ModelNet40/curtain/train/curtain_0130.off" "$DATASET_ROOT/geo/USACensusBlockGroupBoundaries.wkt" "$DATASET_ROOT/graphics/happy_buddha.ply")

  for i in {0..3}; do
    dataset_label=${dataset_labels[$i]}
    log="${log_dir}/run_all/rt_analysis/hit_count/${dataset_label}.json"

    echo "${log}" | xargs dirname | xargs mkdir -p

    if [[ -f "$log" ]]; then
      echo "Skipping, $log exists"
    else
      $PROG_ROOT/hd_exec \
        -input1 "${datasets1[$i]}" \
        -input2 "${datasets2[$i]}" \
        -input_type ${dataset_types[$i]} \
        -n_dims ${dataset_dims[$i]} \
        -serialize $SERIALIZE_ROOT \
        -variant "compare-methods" \
        -execution "gpu" \
        -repeat 1 \
        -json "$log" \
        -check=false \
        -normalize="${norm_flags[$i]}" \
        -max_hit=999999999
    fi
  done
}

function run_analysis_max_hit_progress() {
  dataset_labels=("MRI" "CAD" "Geospatial" "Graphics")
  dataset_types=("image" "off" "wkt" "ply")
  dataset_dims=(3 3 2 3)
  norm_flags=("false" "true" "false" "false")
  datasets1=("$DATASET_ROOT/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_011/BraTS20_Validation_011_flair.nii" "$DATASET_ROOT/ModelNet40/airplane/train/airplane_0169.off" "$DATASET_ROOT/geo/USADetailedWaterBodies.wkt" "$DATASET_ROOT/graphics/dragon.ply")
  datasets2=("$DATASET_ROOT/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_092/BraTS20_Validation_092_t1ce.nii" "$DATASET_ROOT/ModelNet40/airplane/train/airplane_0130.off" "$DATASET_ROOT/geo/USACensusBlockGroupBoundaries.wkt" "$DATASET_ROOT/graphics/happy_buddha.ply")

  for i in {0..3}; do
    for ((max_hit = 0; max_hit < 100; max_hit ++)); do
      dataset_label=${dataset_labels[$i]}
      log="${log_dir}/run_all/rt_analysis/max_hit/${dataset_label}_max_hit_${max_hit}.json"

      echo "${log}" | xargs dirname | xargs mkdir -p

      if [[ -f "$log" ]]; then
        echo "Skipping, $log exists"
      else
        $PROG_ROOT/hd_exec \
          -input1 "${datasets1[$i]}" \
          -input2 "${datasets2[$i]}" \
          -input_type ${dataset_types[$i]} \
          -n_dims ${dataset_dims[$i]} \
          -serialize $SERIALIZE_ROOT \
          -variant "compare-methods" \
          -execution "gpu" \
          -repeat 1 \
          -json "$log" \
          -check=false \
          -normalize="${norm_flags[$i]}" \
          -max_hit=$max_hit
      fi
    done
  done
}

#run_analysis_vary_dist
#run_analysis_vary_max_hit
run_analysis_hit_count
run_analysis_max_hit_progress
