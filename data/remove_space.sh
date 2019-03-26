#!/bin/sh
in_file=$1

file_name=${in_file%.*}
ext=${in_file##*.}

sed -e 's/ //g' $in_file > $file_name"-remove-space."${ext}
