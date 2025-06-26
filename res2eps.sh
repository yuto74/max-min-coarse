#!/usr/bin/sh

for file in `ls ./* | grep \\\.res`
do
    out=`echo $file | sed 's/\.res/.eps/g'`
    echo $out
    xdoplot -te $file | psfix-gnuplot > $out
done
