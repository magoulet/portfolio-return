#!/bin/bash

d=2021-02-19
while [ "$d" != 2021-05-24 ]; do
  echo $d
  python3 Portfolio_Return.py -d $d -o
  d=$(date -I -d "$d + 1 day")
  echo "------------------------"
done
