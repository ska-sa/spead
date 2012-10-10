#!/bin/bash

echo "set term x11 size 1200,720"
echo "set style data line"
echo "set style line 1 lt 2 lw 2 pt 0 lc rgb \"green\""
echo "set style line 2 lt 2 lw 2 pt 0 lc rgb \"red\""

echo "plot \"-\" ls 1 title \"power\", \"-\" ls 2 title \"phase\""


while read l
do
  echo $l
done

sleep  100


