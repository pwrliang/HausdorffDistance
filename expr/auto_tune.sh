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

  name1=$(basename $input1)
  name2=$(basename $input2)

  log="${log_dir}/${out_prefix}/${name1}_${name2}"

  echo "${log}" | xargs dirname | xargs mkdir -p

  $PROG_ROOT/hd_exec \
    -input1 $input1 \
    -input2 $input2 \
    -input_type $input_type \
    -n_dims $n_dims \
    -serialize $SERIALIZE_ROOT \
    -autotune \
    -n_points_cell_list "1,2,4,8,16,32" \
    -max_hit_list "1,2,4,8,16,32,64,128,256" \
    -max_hit_reduce_factor_list "1" \
    -radius_step_list "1.2,1.4,1.6,1.8,2.0" \
    -sample_rate_list "0.0001,0.0005,0.001,0.005,0.01" \
    -check=false \
    -repeat 3 \
    -json "$log"
}

function run_datasets() {
  root=$1
  type=$2
  dims=$3

  if [[ -f "$root/.list" ]]; then
    list=$(cat "$root/.list")
  else
    list=$(find "$root" -type f)
    list=$(echo "$list")
    echo "$list" >"$root/.list"
  fi

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

      vary_variables "$out_prefix" "$file1" "$file2" $type $dims
    else
      echo "Error: Could not pick two files. Skipping iteration."
      ((counter += 2)) # Still increment to avoid infinite loop
    fi
  done
}

run_datasets "/local/storage/shared/BraTS2020_TrainingData" "image" 3
