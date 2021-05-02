#!/bin/bash

d=2016-04-01
while [ "$d" != 2020-09-16 ]; do
  echo $d
  python3 Portfolio_Return.py -d $d -t transactions.csv -o results.csv
  d=$(date -I -d "$d + 1 day")
  echo "------------------------"
done
