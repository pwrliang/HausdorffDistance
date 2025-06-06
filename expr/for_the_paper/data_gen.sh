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

source "${script_dir}/../common.sh"

for dist in uniform gaussian sierpinski bit; do
  for size in 10000000 20000000 30000000 40000000 50000000 60000000; do
    for seed in 1 2; do
      out_path="$DATASET_ROOT/synthetic/${dist}_seed_${seed}_n_${size}.wkt"
      if [[ ! -f "$out_path" ]]; then
        python3 generator.py \
          distribution=$dist \
          cardinality=$size \
          probability=0.2 \
          digits=10 \
          seed=$seed \
          dimensions=3 \
          geometry=point \
          format=wkt \
          affinematrix=1,0,0,0,1,0 \
          affinematrix=1,0,0,0,1,0 >"$out_path"
      fi
    done
  done
done
