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

  log="${log_dir}/train/${out_prefix}/${name1}_${name2}"

  echo "${log}" | xargs dirname | xargs mkdir -p

  $PROG_ROOT/hd_exec \
    -input1 $input1 \
    -input2 $input2 \
    -input_type $input_type \
    -n_dims $n_dims \
    -serialize $SERIALIZE_ROOT \
    -vary_params \
    -n_points_cell_list "1,2,4,6,8,10,12,14,16,18,20,22,24,30,60,120,240" \
    -repeat 5 \
    -json "$log"

  $PROG_ROOT/hd_exec \
    -input1 $input1 \
    -input2 $input2 \
    -input_type $input_type \
    -n_dims $n_dims \
    -serialize $SERIALIZE_ROOT \
    -vary_params \
    -sample_rate_list "0.0001,0.0005,0.001,0.005,0.01" \
    -repeat 5 \
    -json "$log"
}

function run_all_datasets() {
  root=$1
  type=$2
  dims=$3

  if [[ -f "$root/.list" ]]; then
    list=$(cat "$root/.list")
  else
    list=$(find "$root" -type f)
    list=$(echo "$list" | shuf)
    echo "$list" >"$root/.list"
  fi

  out_prefix=$(basename "$root")
  FILE_LIMIT=1000
  CURR_FILE_IDX=0
  while IFS= read -r file1; do
    while IFS= read -r file2; do
      if [[ "$file1" != "$file2" ]]; then
        CURR_FILE_IDX=$((CURR_FILE_IDX + 1))
        echo "file $CURR_FILE_IDX: $file1 $file2"

        vary_variables_independent "$out_prefix" "$file1" "$file2" $type $dims
        if [[ $CURR_FILE_IDX -ge $FILE_LIMIT ]]; then
          return
        fi
      fi
    done < <(printf '%s\n' "$list")
  done < <(printf '%s\n' "$list")
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

run_datasets "/local/storage/shared/HDDatasets/BraTS2020_TrainingData" "image" 3
run_datasets "/local/storage/shared/HDDatasets/ModelNet40" "off" 3
#run_all_datasets "/local/storage/shared/hd_datasets" "wkt" 2
