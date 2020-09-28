


from tfrecord_utils.encoders import TFRecordCoder as BaseTFRecordCoder




class TFRecordCoder(BaseTFRecordCoder):

    def encode_example(self, example : dict):
        img, label, path = example['image'], example['label'], example['path']
        img = tf.image.encode_jpeg(img, optimize_size=True, chroma_downsampling=False)

        features = {
                    'image': bytes_feature(img),
                    'label': int64_feature(label),
                    'path': bytes_feature(tf.compat.as_bytes(path))
                    }
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()

    def decode_example(self, example):
        feature_description = {
                                'image': tf.io.FixedLenFeature([], tf.string),
                                'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
                                'path': tf.io.FixedLenFeature([], tf.string)
                                }
        features = tf.io.parse_single_example(example,features=feature_description)

        img = tf.image.decode_jpeg(features['image'], channels=3) # * 255.0
        img = tf.image.resize_image_with_pad(img, *self.target_shape[:2])

        label = tf.cast(features['label'], tf.int32)
        label = tf.one_hot(label, depth=self.num_classes)
        return img, label