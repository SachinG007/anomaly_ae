#!/bin/bash
for nrad in 1.2 1.4 1.6 1.8 2 2.2 2.5 3
do
for s in 6
do
for lr in 0.0001  
do
   CUDA_VISIBLE_DEVICES="2"  python3 train_sin_ae.py -dev gpu  -nr $nrad -e 50 -lr $lr -s $s &
   CUDA_VISIBLE_DEVICES="2"  python3 train_sin_ae.py -dev gpu  -nr $nrad -e 50 -lr $lr -s $s &
   CUDA_VISIBLE_DEVICES="2"  python3 train_sin_ae.py -dev gpu  -nr $nrad -e 50 -lr $lr -s $s &
   CUDA_VISIBLE_DEVICES="2"  python3 train_sin_ae.py -dev gpu  -nr $nrad -e 50 -lr $lr -s $s &
   CUDA_VISIBLE_DEVICES="2"  python3 train_sin_ae.py -dev gpu  -nr $nrad -e 50 -lr $lr -s $s &
   CUDA_VISIBLE_DEVICES="2"  python3 train_sin_ae.py -dev gpu  -nr $nrad -e 50 -lr $lr -s $s
#    echo $nrad
done
done
done
# for nrad in 1 1.2 1.5 2 10
# do
#    CUDA_VISIBLE_DEVICES="3"  python3 train_sin_ae.py -dev gpu  -nr $nrad -e 50 -lr 0.001 -s 6 &
#    CUDA_VISIBLE_DEVICES="3"  python3 train_sin_ae.py -dev gpu  -nr $nrad -e 50 -lr 0.001 -s 6 &
#    CUDA_VISIBLE_DEVICES="3"  python3 train_sin_ae.py -dev gpu  -nr $nrad -e 50 -lr 0.001 -s 6 &
#    CUDA_VISIBLE_DEVICES="3"  python3 train_sin_ae.py -dev gpu  -nr $nrad -e 50 -lr 0.001 -s 6 &
#    CUDA_VISIBLE_DEVICES="3"  python3 train_sin_ae.py -dev gpu  -nr $nrad -e 50 -lr 0.001 -s 6
# #    echo $nrad
# done