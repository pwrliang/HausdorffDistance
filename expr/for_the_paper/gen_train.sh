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

function vary_variables() {
  out_prefix=$1
  input1=$2
  input2=$3
  input_type=$4
  n_dims=$5
  normalize=$6

  name1=$(basename $input1)
  name2=$(basename $input2)

  log="${log_dir}/train/${out_prefix}/eb_only_threshold/${name1}_${name2}"
  echo "${log}" | xargs dirname | xargs mkdir -p

  $PROG_ROOT/hd_exec \
    -input1 $input1 \
    -input2 $input2 \
    -input_type $input_type \
    -n_dims $n_dims \
    -serialize $SERIALIZE_ROOT \
    -vary_params \
    -eb_only_threshold_list "1,100,200,300,400,500,600,700,800,900,1000" \
    -repeat 5 \
    -json "$log" \
    -normalize="$normalize"

  log="${log_dir}/train/${out_prefix}/n_points_cell/${name1}_${name2}"
  echo "${log}" | xargs dirname | xargs mkdir -p

  $PROG_ROOT/hd_exec \
    -input1 $input1 \
    -input2 $input2 \
    -input_type $input_type \
    -n_dims $n_dims \
    -serialize $SERIALIZE_ROOT \
    -vary_params \
    -n_points_cell_list "1,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80" \
    -repeat 5 \
    -json "$log" \
    -normalize="$normalize"

  log="${log_dir}/train/${out_prefix}/max_hit/${name1}_${name2}"
  echo "${log}" | xargs dirname | xargs mkdir -p

  $PROG_ROOT/hd_exec \
    -input1 $input1 \
    -input2 $input2 \
    -input_type $input_type \
    -n_dims $n_dims \
    -serialize $SERIALIZE_ROOT \
    -vary_params \
    -max_hit_list "1,16,32,64,128,256,512" \
    -repeat 5 \
    -json "$log" \
    -normalize="$normalize"
}

function run_mri_datasets() {
  root=$1
  type=$2
  dims=$3

  list=$(find "$root" -type f | grep nii)
  list=$(echo "$list")

  mapfile -t files <<<"$list"

  file_count=${#files[@]}
  if ((file_count < 2)); then
    echo "Not enough files in the directory!"
    exit 1
  fi
  seed=42
  counter=0
  max=1000 # Total files to pick (2 per iteration = 500 loops)
  out_prefix=$(basename "$root")

  while ((counter < max)); do
    current_seed="${seed}_${counter}"

    # Use openssl to generate a stream and feed it directly to shuf
    mapfile -t picks < <(
      printf "%s\n" "${files[@]}" |
        shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"$current_seed" -nosalt </dev/zero 2>/dev/null) |
        head -n 2
    )
    file1="${picks[0]}"
    file2="${picks[1]}"
    # Ensure we got two valid files
    if [[ -n "$file1" && -n "$file2" ]]; then
      echo "Pick $((counter + 1)): $file1"
      ((counter++))
      echo "Pick $((counter + 1)): $file2"
      ((counter++))

      if [[ "$file1" != "$file2" ]]; then
        vary_variables "$out_prefix" "$file1" "$file2" $type $dims "false"
      else
        ((counter -= 2))
      fi
    else
      echo "Error: Could not pick two files. Skipping iteration."
      ((counter += 2)) # Still increment to avoid infinite loop
    fi
  done
}

function run_modelnet_datasets() {
  root=$1
  type=$2
  dims=$3

  list=$(find "$root" -type f | grep train)
  list=$(echo "$list")

  mapfile -t files <<<"$list"

  file_count=${#files[@]}
  if ((file_count < 2)); then
    echo "Not enough files in the directory!"
    exit 1
  fi
  seed=42
  counter=0
  max=200 # Total files to pick (2 per iteration = 500 loops)
  out_prefix=$(basename "$root")

  while ((counter < max)); do
    current_seed="${seed}_${counter}"
    # Use openssl to generate a stream and feed it directly to shuf
    mapfile -t picks < <(
      printf "%s\n" "${files[@]}" |
        shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"$current_seed" -nosalt </dev/zero 2>/dev/null) |
        head -n 2
    )
    file1="${picks[0]}"
    file2="${picks[1]}"
    # Ensure we got two valid files
    if [[ -n "$file1" && -n "$file2" ]]; then
      echo "Pick $((counter + 1)): $file1"
      ((counter++))
      echo "Pick $((counter + 1)): $file2"
      ((counter++))

      name1=$(basename $file1)
      name2=$(basename $file2)

      if [[ "$file1" != "$file2" ]]; then
        vary_variables "$out_prefix" "$file1" "$file2" $type $dims "true"
      else
        ((counter -= 2))
      fi
    else
      echo "Error: Could not pick two files. Skipping iteration."
      ((counter += 2)) # Still increment to avoid infinite loop
    fi
  done
}

function run_geo_datasets() {
  root=$1
  type=$2
  dims=$3

  for file1 in "$root"/*.wkt; do
    for file2 in "$root"/*.wkt; do
      if [[ "$file1" != "$file2" ]]; then
        vary_variables "geo" "$file1" "$file2" $type $dims "false"
      fi
    done
  done
}

run_mri_datasets "$DATASET_ROOT/BraTS2020_TrainingData" "image" 3
run_modelnet_datasets "$DATASET_ROOT/ModelNet40" "off" 3
run_geo_datasets "$DATASET_ROOT/geo/train" "wkt" 2
