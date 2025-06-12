#!/usr/bin/sh

for file in `ls * |grep degree`
do
    out=`echo $file | sed 's/degree/r2/g'`
    mv $file $out
done
