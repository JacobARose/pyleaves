#Run this script to generate all the datasets as TFRecords in one go

echo "Generating PNAS TFRecords and outputing to /media/data/jacob/PNAS"
python create_tfrecords.py --dataset_name PNAS --output_dir /media/data/jacob -thresh 3 -val 0.3 -test 0.3

# python create_tfrecords.py --dataset_name Leaves --output_dir /media/data/jacob -thresh 3 -val 0.3 -test 0.3


# python create_tfrecords.py --dataset_name Fossil --output_dir /media/data/jacob -thresh 3 -val 0.3 -test 0.3