from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from scipy.io import savemat
from ops import *


class NRDS(object):
    def __init__(self,
                 session,  # TensorFlow session
                 size_image=128,  # size the input images
                 size_kernel=5,  # size of the kernels in convolution and deconvolution
                 size_batch=1,  # mini-batch size for training and testing, must be square of an integer
                 num_input_channels=3,  # number of channels of input images
                 enable_bn=True,  # enable batch normalization
                 save_dir='./save',  # path to save checkpoints, samples, and summary
                 real_files_dir=None,  # directory of the real data
                 fake_files_dirs=None  # directories of the fake data
                 ):

        self.session = session
        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_input_channels = num_input_channels
        self.enable_bn = enable_bn
        self.save_dir = save_dir
        self.real_files_dir = real_files_dir
        self.fake_files_dirs = fake_files_dirs
        if self.real_files_dir is None or self.fake_files_dirs is None:
            print('Missing real or fake samples!')
            exit()

        # ************************************* input to graph ********************************************************
        self.real = tf.placeholder(
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
            name='input_real'
        )
        self.fake = tf.placeholder(
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
            name='input_fake'
        )

        # ************************************* build the graph *******************************************************
        # discriminator on real images
        self.D_real, self.D_real_logits = self.discriminator(
            image=self.real,
            enable_bn=self.enable_bn,
            reuse_variables=False
        )
        # discriminator on real image
        self.D_fake, self.D_fake_logits = self.discriminator(
            image=self.fake,
            enable_bn=self.enable_bn,
            reuse_variables=True
        )

        # ************************************* loss functions *******************************************************
        self.loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_real_logits, tf.ones_like(self.D_real_logits))
        )
        self.loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_logits, tf.zeros_like(self.D_fake_logits))
        )

        # ************************************* collect the summary ***************************************
        self.loss_real_summary = tf.summary.scalar('real', self.loss_real)
        self.loss_fake_summary = tf.summary.scalar('fake', self.loss_fake)
        self.summary = tf.summary.merge([
            self.loss_real_summary, self.loss_fake_summary
        ])
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def train(self,
              num_epochs=50,  # number of epochs
              learning_rate=0.0002,  # learning rate of optimizer
              beta1=0.5,  # parameter for Adam optimizer
              ):
        # *************************** load file names of images ******************************************************
        real_files = glob(os.path.join(self.real_files_dir, '*.jpg'))
        print("real samples:\t%d" % len(real_files))
        fake_files_separate = []
        fake_files = []
        for i, fake_dir in enumerate(self.fake_files_dirs):
            fake_files_separate.append(glob(os.path.join(fake_dir, '*.jpg')))
            fake_files.extend(glob(os.path.join(fake_dir, '*.jpg')))
            print("%s samples:\t%d" % (self.fake_files_dirs[i].split('/')[-1], len(fake_files_separate[i])))

        # *********************************** optimizer **************************************************************
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_real + self.loss_fake,
        )

        # ******************************************* training *******************************************************
        tf.global_variables_initializer().run()
        mat_save_path = os.path.join(self.save_dir, 'mat')
        np.random.seed(seed=1234)
        if not os.path.exists(mat_save_path):
            os.makedirs(mat_save_path)
        for epoch in xrange(num_epochs):
            np.random.shuffle(real_files)
            np.random.shuffle(fake_files)
            num_batches = min(len(real_files), len(fake_files)) // self.size_batch
            for ind_batch in range(0, num_batches):
                real_files_batch = real_files[ind_batch * self.size_batch: (ind_batch + 1) * self.size_batch]
                fake_files_batch = fake_files[ind_batch * self.size_batch: (ind_batch + 1) * self.size_batch]
                real_images_batch = [
                    load_image(
                        image_path=batch_file,
                        image_size=self.size_image,
                        image_value_range=self.image_value_range,
                        is_gray=(self.num_input_channels == 1),
                    ) for batch_file in real_files_batch]
                fake_images_batch = [
                    load_image(
                        image_path=batch_file,
                        image_size=self.size_image,
                        image_value_range=self.image_value_range,
                        is_gray=self.num_input_channels == 1,
                    ) for batch_file in fake_files_batch]
                if self.num_input_channels == 1:
                    batch_images_real = np.array(real_images_batch).astype(np.float32)[:, :, :, None]
                    batch_images_fake = np.array(fake_images_batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images_real = np.array(real_images_batch).astype(np.float32)
                    batch_images_fake = np.array(fake_images_batch).astype(np.float32)

                # Update the discriminator
                _, loss_real, loss_fake, summary = self.session.run(
                    fetches=[optimizer, self.loss_real, self.loss_fake, self.summary],
                    feed_dict={
                        self.real: batch_images_real,
                        self.fake: batch_images_fake
                    }
                )
                self.writer.add_summary(summary, self.global_step.eval())
                print("\nEpoch %03d/%03d\tBatch %03d/%03d\t loss_real=%.4f\t loss_fake=%.4f\n"
                      % (epoch+1, num_epochs, ind_batch+1, num_batches, loss_real, loss_fake))

            # testing on fake data
            mat = dict()
            mat['epochs'] = num_epochs
            mat['batches'] = num_batches
            num_competitors = len(fake_files_separate)
            for i in range(num_competitors):
                files = fake_files_separate[i]
                folder_name = self.fake_files_dirs[i].split('/')[-1]
                mat[folder_name] = []

                for ind in range(0, len(files) // self.size_batch):
                    files_batch = files[ind * self.size_batch: (ind + 1) * self.size_batch]
                    images_batch = [
                        load_image(
                            image_path=batch_file,
                            image_size=self.size_image,
                            image_value_range=self.image_value_range,
                            is_gray=(self.num_input_channels == 1),
                        ) for batch_file in files_batch]
                    if self.num_input_channels == 1:
                        images_batch = np.array(images_batch).astype(np.float32)[:, :, :, None]
                    else:
                        images_batch = np.array(images_batch).astype(np.float32)
                    output = self.D_fake.eval({self.fake: images_batch})
                    mat[folder_name].append(output)

            # testing on real data
            mat['real'] = []
            for ind in xrange(0, len(real_files) // self.size_batch):
                files_batch = real_files[ind * self.size_batch: (ind + 1) * self.size_batch]
                images_batch = [
                    load_image(
                        image_path=batch_file,
                        image_size=self.size_image,
                        image_value_range=self.image_value_range,
                        is_gray=(self.num_input_channels == 1),
                    ) for batch_file in files_batch]
                if self.num_input_channels == 1:
                    images_batch = np.array(images_batch).astype(np.float32)[:, :, :, None]
                else:
                    images_batch = np.array(images_batch).astype(np.float32)
                output = self.D_real.eval({self.real: images_batch})
                mat['real'].append(output)

            savemat(os.path.join(mat_save_path, '{:03d}_{:03d}.mat'.format(epoch + 1, ind_batch + 1)), mat)

    def discriminator(self, image, is_training=True, reuse_variables=False, enable_bn=True,
                      num_hidden_layer_channels=(64, 128, 256, 512)):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = len(num_hidden_layer_channels)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'D_img_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=num_hidden_layer_channels[i],
                    size_kernel=self.size_kernel,
                    name=name
                )
            if enable_bn:
                name = 'D_img_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)
        # fully connection layer
        name = 'D_img_fc1'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=1024,
            name=name
        )
        current = lrelu(current)
        name = 'D_img_fc2'
        current = fc(
            input_vector=current,
            num_output_length=1,
            name=name
        )
        # output
        return tf.nn.sigmoid(current), current



