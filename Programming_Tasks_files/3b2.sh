##!/bin/bash
for file in Zalora-*
do
    uppercasetext="$(awk '{print toupper($0)}' < $file)"
    echo $uppercasetext > $file
done