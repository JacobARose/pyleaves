#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python '/home/jacob/projects/pyleaves/pyleaves/mains/train_parallel.py'  +fold_id=0 stage_0.misc.use_tfrecords=False &

CUDA_VISIBLE_DEVICES=1 python '/home/jacob/projects/pyleaves/pyleaves/mains/train_parallel.py'  +fold_id=1 stage_0.misc.use_tfrecords=False &

CUDA_VISIBLE_DEVICES=2 python '/home/jacob/projects/pyleaves/pyleaves/mains/train_parallel.py'  +fold_id=2 stage_0.misc.use_tfrecords=False