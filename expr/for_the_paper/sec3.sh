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
  auto_tune=false
  if [[ $variant == "hybrid" ]]; then
    auto_tune=true
  fi

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
      -v=1 \
      -auto_tune="$auto_tune"
  fi
}

function run_mri_datasets() {
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

function run_modelnet_datasets() {
  root=$1
  variant=$2
  execution=$3

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
  out_prefix=$(basename "$root")
  max=50 # Total files to pick per folder

  for sub_folder in "$root"/*/; do
    list=$(find "$sub_folder" -type f -exec stat --format="%s %n" {} + | grep '.off' | sort -nr | awk '{ $1=""; sub(/^ /, ""); print }' | head -n $max)
    mapfile -t files <<<"$list"

    file_count=${#files[@]}
    if ((file_count < 2)); then
      echo "Not enough files in the directory!"
      exit 1
    fi
    seed=42
    counter=0

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

        log="${log_dir}/run_all/${variant}_${execution}/${out_prefix}/${name1}_${name2}.json"

        echo "${log}" | xargs dirname | xargs mkdir -p

        if [[ -f "$log" ]]; then
          echo "Skipping, $log exists"
        else
          auto_tune=false
          if [[ $variant == "hybrid" ]]; then
            auto_tune=true
          fi
          $PROG_ROOT/hd_exec \
            -input1 "$file1" \
            -input2 "$file2" \
            -input_type "off" \
            -n_dims 3 \
            -serialize $SERIALIZE_ROOT \
            -variant $variant \
            -execution $execution \
            -repeat 5 \
            -json "$log" \
            -check=false \
            -auto_tune="$auto_tune" \
            -normalize # need to normalize data to the same coordinate system
        fi
      else
        echo "Error: Could not pick two files. Skipping iteration."
        ((counter += 2)) # Still increment to avoid infinite loop
      fi
    done
  done
}

function run_mri() {
  for variant in eb rt hybrid; do
    run_mri_datasets "/local/storage/shared/HDDatasets/BraTS2020_TrainingData" "image" 3 $variant "gpu"
  done
}

#run_mri

function run_modelnet() {
  for variant in eb rt hybrid; do #
    run_modelnet_datasets "/local/storage/shared/HDDatasets/ModelNet40" $variant "gpu"
  done
}

#run_modelnet

function run_geo() {
  root="/local/storage/shared/HDDatasets/geo"
  datasets1=(dtl_cnty.wkt USADetailedWaterBodies.wkt lakes.bz2.wkt)
  datasets2=(uszipcode.wkt USACensusBlockGroupBoundaries.wkt parks.bz2.wkt)
  for ((i = 0; i < ${#datasets1[@]}; i++)); do
    dataset1=${datasets1[i]}
    dataset2=${datasets2[i]}
    for variant in eb rt hybrid; do #
      run_hd "geo" "$root/$dataset2" "$root/$dataset1" "wkt" 2 $variant "gpu"
    done
  done
}

run_geo
