##!/bin/bash
for file in zalora-*
do 
    mv "$file" Z"${file#z}"
done