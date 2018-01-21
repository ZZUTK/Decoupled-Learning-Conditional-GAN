import tensorflow as tf
from NRDS import NRDS
import checkGPU
import numpy as np


flags = tf.app.flags
flags.DEFINE_integer(flag_name='epoch', default_value=100, docstring='number of epochs')
flags.DEFINE_integer(flag_name='batch_size', default_value=1, docstring='batch size')
flags.DEFINE_integer(flag_name='is_bn', default_value=0, docstring='enable batch normalization')
flags.DEFINE_string(flag_name='save_dir', default_value='save', docstring='dir for saving training results')
flags.DEFINE_integer(flag_name='is_schedule', default_value=0, docstring='scheduled running')
flags.DEFINE_integer(flag_name='day', default_value=1, docstring='date')
flags.DEFINE_integer(flag_name='hr', default_value=0, docstring='hour')
flags.DEFINE_integer(flag_name='min', default_value=0, docstring='minute')

FLAGS = flags.FLAGS

gpu_memory_require = 5.0


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
        model = NRDS(
            session,  # TensorFlow session
            save_dir=FLAGS.save_dir,  # path to save checkpoints, samples, and summary
            size_batch=FLAGS.batch_size,
            enable_bn=FLAGS.is_bn,
            real_files_dir='./results/real',
            fake_files_dirs=[
                './results/adv1',
                './results/adv1e2',
                './results/adv1e3',
                './results/adv1e4',
            ]
        )
        model.train(
            num_epochs=FLAGS.epoch,  # number of epochs
        )


if __name__ == '__main__':
    if 0:
        print 'Run on CPU'
        with tf.device("/cpu:0"):
            gpu_memory_require = 0.0
            tf.app.run()

    tf.app.run()

