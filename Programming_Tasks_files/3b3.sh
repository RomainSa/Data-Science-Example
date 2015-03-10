##!/bin/bash
for file in Zalora-*
do
    nodottext="$(tr -d "." < $file)"
    echo $nodottext > $file
done