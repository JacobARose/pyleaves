'''
Test functions related to tensorflow 

'''
from pyleaves.leavesdb.tf_utils.create_tfrecords import load_and_encode_example, decode_example

def test_load_and_encode_example():
    dummy_sample = {'path':r'/media/data_cifs/sven2/leaves/sorted/Fossils_DataSource/New_Fossil_Dataset/I. Approved families/Adoxaceae/Sambucus newtoni/CU_0141 Sambucus newtoni.tif',
                   'label':0}
    serialized_example = load_and_encode_example(**dummy_sample)
    img, label = decode_example(serialized_example)
    return img, label