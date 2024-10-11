#!/bin/bash

for i in {1..4}
do
   python ./prehoc/train_dynemo.py 3 $i
   # python ./prehoc/train_dynemo.py 4 $i
   # python ./prehoc/train_dynemo.py 5 $i
   # python ./prehoc/train_dynemo.py 6 $i
done
