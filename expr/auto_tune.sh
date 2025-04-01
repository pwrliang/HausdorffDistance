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
    -max_hit_reduce_factor_list "1,1.2,1.4,1.6,1.8,2" \
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

        vary_variables "$out_prefix" "$file1" "$file2" $type $dims
        if [[ $CURR_FILE_IDX -ge $FILE_LIMIT ]]; then
          return
        fi
      fi
    done < <(printf '%s\n' "$list")
  done < <(printf '%s\n' "$list")
}

run_datasets "/local/storage/shared/BraTS2020_TrainingData" "image" 3
