#!/bin/bash

echo "set title 'heat map'"
echo "unset key"
echo "set tic scale 0"
echo "set view map"

while read l
  do
    echo $l
  done
