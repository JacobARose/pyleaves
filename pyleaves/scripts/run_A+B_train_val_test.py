# @Author: Jacob A Rose
# @Date:   Tue, April 28th 2020, 1:35 am
# @Email:  jacobrose@brown.edu
# @Filename: run_A+B_train_val_test.sh

python /home/jacob/projects/pyleaves_git_correction/pyleaves/pyleaves/mains/main_2-dataset_1-stage_experiment.py \
--experiment_type 'A+B_train_val_test' \
--model 'vgg16' \
--run_id '2000' \
--batch_size 32 \
--learning_rate 3e-4 \
--num_epochs 100 \
--gpu '2' \
--seed 10293

# python /home/jacob/projects/pyleaves_git_correction/pyleaves/pyleaves/mains/main_single_dataset_experiment.py \
# --experiment_type 'A+B_train_val_test' \
# --dataset_A 'PNAS'
# --dataset_B 'Fossil'
# --model 'vgg16'
# --run_id '2000'
# --batch_size 32
# --learning_rate 3e-4
# --num_epochs 100
# --gpu '0'
# --seed 10293



tfrecord_paths
['/media/data/jacob/Fossil_Project/data/tfrecord_data/A+B_train_val_test/PNAS+Fossil/train-00000-of-00010.tfrecord', '/media/data/jacob/Fossil_Project/data/tfrecord_data/A+B_train_val_test/PNAS+Fossil/train-00001-of-00010.tfrecord', '/media/data/jacob/Fossil_Project/data/tfrecord_data/A+B_train_val_test/PNAS+Fossil/train-00002-of-00010.tfrecord', '/media/data/jacob/Fossil_Project/data/tfrecord_data/A+B_train_val_test/PNAS+Fossil/train-00003-of-00010.tfrecord', '/media/data/jacob/Fossil_Project/data/tfrecord_data/A+B_train_val_test/PNAS+Fossil/train-00004-of-00010.tfrecord', '/media/data/jacob/Fossil_Project/data/tfrecord_data/A+B_train_val_test/PNAS+Fossil/train-00005-of-00010.tfrecord', '/media/data/jacob/Fossil_Project/data/tfrecord_data/A+B_train_val_test/PNAS+Fossil/train-00006-of-00010.tfrecord', '/media/data/jacob/Fossil_Project/data/tfrecord_data/A+B_train_val_test/PNAS+Fossil/train-00007-of-00010.tfrecord', '/media/data/jacob/Fossil_Project/data/tfrecord_data/A+B_train_val_test/PNAS+Fossil/train-00008-of-00010.tfrecord', '/media/data/jacob/Fossil_Project/data/tfrecord_data/A+B_train_val_test/PNAS+Fossil/train-00009-of-00010.tfrecord', '/media/data/jacob/Fossil_Project/data/tfrecord_data/A+B_train_val_test/PNAS+Fossil/train-00010-of-00010.tfrecord']
