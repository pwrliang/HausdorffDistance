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

  name1=$(basename $input1)
  name2=$(basename $input2)

  log="${log_dir}/run_all/${variant}_${execution}/${out_prefix}/${name1}_${name2}.json"

  echo "${log}" | xargs dirname | xargs mkdir -p

  if [[ -f "$log" ]]; then
    echo "Skipping, $log exists"
  else
    if [[ $variant == "monai" ]]; then
      python3 run_monai.py $input1 $input2 $log
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
        -check=false
    fi
  fi
}

function run_datasets() {
  root=$1
  type=$2
  dims=$3
  variant=$4
  execution=$5

  list=$(find "$root" -type f)
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

      run_hd "$out_prefix" "$file1" "$file2" $type $dims $variant $execution
    else
      echo "Error: Could not pick two files. Skipping iteration."
      ((counter += 2)) # Still increment to avoid infinite loop
    fi
  done
}

function run_same_type_datasets() {
  root=$1
  keyword=$2
  type=$3
  dims=$4
  variant=$5
  execution=$6

  for sub_folder in "$root"/*/; do
    list=$(find "$sub_folder" -type f | grep "$keyword")

    mapfile -t files <<<"$list"

    file_count=${#files[@]}
    if ((file_count < 2)); then
      echo "Not enough files in the directory!"
      exit 1
    fi
    seed=42
    counter=0
    max=20 # Total files to pick (2 per iteration = 500 loops)
    out_prefix=$(basename "$sub_folder")

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
        run_hd "$out_prefix" "$file1" "$file2" $type $dims $variant $execution
      else
        echo "Error: Could not pick two files. Skipping iteration."
        ((counter += 2)) # Still increment to avoid infinite loop
      fi
    done
  done
}

function compare_eb() {
  run_datasets "/local/storage/shared/HDDatasets/BraTS2020_ValidationData" "image" 3 "eb" "gpu"
  run_datasets "/local/storage/shared/HDDatasets/BraTS2020_ValidationData" "image" 3 "itk" "cpu"
  run_datasets "/local/storage/shared/HDDatasets/BraTS2020_ValidationData" "image" 3 "monai" "cpu"
}

function compare_rt() {
  dataset1=(dragon.ply thai_statuette.ply dragon.ply thai_statuette.ply)
  dataset2=(asian_dragon.ply happy_buddha.ply happy_buddha.ply asian_dragon.ply)
  bit_counts=(7 7 6 7)
  dataset_root="/local/storage/shared/HDDatasets/graphics"
  for ((i = 0; i < ${#dataset1[@]}; i++)); do
    name1=${dataset1[i]}
    name2=${dataset2[i]}
    bit_count=${bit_counts[i]}
    input1="${dataset_root}/${name1}"
    input2="${dataset_root}/${name2}"

    log="${log_dir}/run_all/compare_rt/rt_hdist/${name1}_${name2}.json"
    echo "${log}" | xargs dirname | xargs mkdir -p

    if [[ ! -f "$log" ]]; then
      $PROG_ROOT/hd_exec \
        -input1 $input1 \
        -input2 $input2 \
        -input_type ply \
        -n_dims 3 \
        -serialize $SERIALIZE_ROOT \
        -variant rt-hdist \
        -execution gpu \
        -repeat 5 \
        -json "$log" \
        -check=false \
        -v=1 \
        -bit_count=$bit_count
    fi

    log="${log_dir}/run_all/compare_rt/rt_grid/${name1}_${name2}.json"
    echo "${log}" | xargs dirname | xargs mkdir -p

    if [[ ! -f "$log" ]]; then
      $PROG_ROOT/hd_exec \
        -input1 $input1 \
        -input2 $input2 \
        -input_type ply \
        -n_dims 3 \
        -serialize $SERIALIZE_ROOT \
        -variant rt \
        -execution gpu \
        -repeat 5 \
        -json "$log" \
        -check=false \
        -v=1 \
        -rt_prune=false \
        -rt_eb=false
    fi

    log="${log_dir}/run_all/compare_rt/rt_prune/${name1}_${name2}.json"
    echo "${log}" | xargs dirname | xargs mkdir -p

    if [[ ! -f "$log" ]]; then
      $PROG_ROOT/hd_exec \
        -input1 $input1 \
        -input2 $input2 \
        -input_type ply \
        -n_dims 3 \
        -serialize $SERIALIZE_ROOT \
        -variant rt \
        -execution gpu \
        -repeat 5 \
        -json "$log" \
        -check=false \
        -v=1 \
        -rt_prune=true \
        -rt_eb=false
    fi

    log="${log_dir}/run_all/compare_rt/rt_eb/${name1}_${name2}.json"
    echo "${log}" | xargs dirname | xargs mkdir -p

    if [[ ! -f "$log" ]]; then
      $PROG_ROOT/hd_exec \
        -input1 $input1 \
        -input2 $input2 \
        -input_type ply \
        -n_dims 3 \
        -serialize $SERIALIZE_ROOT \
        -variant rt \
        -execution gpu \
        -repeat 5 \
        -json "$log" \
        -check=false \
        -v=1 \
        -rt_prune=false \
        -rt_eb=true
    fi

    log="${log_dir}/run_all/compare_rt/rt/${name1}_${name2}.json"
    echo "${log}" | xargs dirname | xargs mkdir -p

    if [[ ! -f "$log" ]]; then
      $PROG_ROOT/hd_exec \
        -input1 $input1 \
        -input2 $input2 \
        -input_type ply \
        -n_dims 3 \
        -serialize $SERIALIZE_ROOT \
        -variant rt \
        -execution gpu \
        -repeat 5 \
        -json "$log" \
        -check=false \
        -v=1 \
        -rt_prune=true \
        -rt_eb=true
    fi
  done
}

compare_eb
#compare_rt

#run_all_datasets "/local/storage/shared/HDDatasets" "wkt" 2

#for variant in eb rt hybrid; do
#  run_same_type_datasets "/local/storage/shared/HDDatasets/ModelNet40" "test" "off" 3 $variant "gpu"
#done
