#!/usr/bin/env bash

set -e

die() { printf "\033[31m$*\033[0m\n" 1>&2 ; exit 1; }

input_dir="$1"
output_dir="$2"
output_dimension=${3:-256}

[ ! -z "$1" ] || die "usage: $0 <input_dir> <output_dir> [output_dimension=256]"

name=`basename "$input_dir"`
final_output_dir="$output_dir"
output_resized_dir="$final_output_dir"_resized
output_edges_dir="$final_output_dir"_edges

mkdir -p "$final_output_dir"
mkdir -p "$output_resized_dir"
mkdir -p "$output_edges_dir"

# resize
for f in $input_dir/*.{jpg,jpeg,png}; do
  convert "$f" -thumbnail "$output_dimension"x"$output_dimension"^ -gravity center -extent "$output_dimension"x"$output_dimension" "$output_resized_dir"/"${f##*/}.png"
done

# detect edges
for f in $output_resized_dir/*.png; do
  python3 auto_canny.py "$f" --out-dir "$output_edges_dir"
done

# combine images into single side-by-side image: edges X resized
for f in $output_resized_dir/*.png; do
  image_filename=$(basename -- "$f")
  convert "$output_edges_dir"/"$image_filename" "$f" +append "$final_output_dir"/"$image_filename"
done
