import tensorflow as tf
from ResLearn import ResLearn
import checkGPU
import numpy as np


flags = tf.app.flags
flags.DEFINE_integer(flag_name='epoch', default_value=50, docstring='number of epochs')
flags.DEFINE_integer(flag_name='batch_size', default_value=25, docstring='batch size')
flags.DEFINE_integer(flag_name='is_train', default_value=1, docstring='training mode')
flags.DEFINE_integer(flag_name='is_bn', default_value=0, docstring='enable batch normalization')
flags.DEFINE_string(flag_name='dataset', default_value='UTKFace', docstring='dataset name')
flags.DEFINE_string(flag_name='savedir', default_value='save', docstring='dir for saving training results')
flags.DEFINE_string(flag_name='testdir', default_value='None', docstring='dir for testing images')
flags.DEFINE_float(flag_name='param0', default_value=1.0, docstring='weight of discriminator loss on fake images')
flags.DEFINE_float(flag_name='param1', default_value=1.0, docstring='weight of reconstruct loss on artifact modeling')
flags.DEFINE_integer(flag_name='is_schedule', default_value=0, docstring='scheduled running')
flags.DEFINE_integer(flag_name='day', default_value=1, docstring='date')
flags.DEFINE_integer(flag_name='hr', default_value=0, docstring='hour')
flags.DEFINE_integer(flag_name='min', default_value=0, docstring='minute')

FLAGS = flags.FLAGS


gpu_memory_require = 7.0


def main(_):
    from datetime import datetime
    if FLAGS.is_schedule:
        today = datetime.today()
        checkGPU.auto_queue(
            gpu_memory_require=gpu_memory_require,
            interval=1,
            schedule=datetime(year=today.year, month=today.month, day=FLAGS.day, hour=FLAGS.hr, minute=FLAGS.min)
        )
    config = checkGPU.set_memory_usage(
        usage=gpu_memory_require,
        allow_growth=True
    )

    # print settings
    import pprint
    pprint.pprint(FLAGS.__flags)

    with tf.Session(config=config) as session:
        model = ResLearn(
            session,  # TensorFlow session
            is_training=FLAGS.is_train,  # flag for training or testing mode
            save_dir=FLAGS.savedir,  # path to save checkpoints, samples, and summary
            dataset_name=FLAGS.dataset,  # name of the dataset in the folder ./data
            size_batch=FLAGS.batch_size,
            enable_bn=FLAGS.is_bn
        )
        if FLAGS.is_train:
            print '\n\tTraining Mode'
            model.train(
                num_epochs=FLAGS.epoch,  # number of epochs
                params=[FLAGS.param0, FLAGS.param1]
            )
        else:
            print '\n\tTesting Mode'
            model.custom_test(
                testing_samples_dir=FLAGS.testdir + '/*jpg'
            )


if __name__ == '__main__':
    if 0:
        print 'Run on CPU'
        with tf.device("/cpu:0"):
            gpu_memory_require = 0.0
            tf.app.run()

    tf.app.run()

