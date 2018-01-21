from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from scipy.io import savemat
from ops import *


class ResLearn(object):
    def __init__(self,
                 session,  # TensorFlow session
                 size_image=128,  # size the input images
                 size_kernel=5,  # size of the kernels in convolution and deconvolution
                 size_batch=100,  # mini-batch size for training and testing, must be square of an integer
                 num_input_channels=3,  # number of channels of input images
                 num_encoder_channels=64,  # number of channels of the first conv layer of encoder
                 num_z_channels=50,  # number of channels of the layer z (noise or code)
                 num_categories=10,  # number of categories (age segments) in the training dataset
                 num_gen_channels=1024,  # number of channels of the first deconv layer of generator
                 enable_tile_label=True,  # enable to tile the label
                 tile_ratio=1.0,  # ratio of the length between tiled label and z
                 is_training=True,  # flag for training or testing mode
                 enable_bn=True,  # enable batch normalization
                 save_dir='./save',  # path to save checkpoints, samples, and summary
                 dataset_name='UTKFace'  # name of the dataset in the folder ./data
                 ):

        self.session = session
        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.num_categories = num_categories
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.enable_bn = enable_bn
        self.save_dir = save_dir
        self.dataset_name = dataset_name

        # ************************************* input to graph ********************************************************
        self.input_image = tf.placeholder(
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
            name='input_images'
        )
        self.age = tf.placeholder(
            tf.float32,
            [self.size_batch, self.num_categories],
            name='age_labels'
        )
        self.gender = tf.placeholder(
            tf.float32,
            [self.size_batch, 2],
            name='gender_labels'
        )
        self.z_prior = tf.placeholder(
            tf.float32,
            [self.size_batch, self.num_z_channels],
            name='z_prior'
        )
        self.params = tf.placeholder(
            tf.float32,
            [2],
            name='params'
        )
        # ************************************* build the graph *******************************************************
        print '\n\tBuilding graph ...'
        # encoder: input image --> z
        self.z = self.encoder(
            image=self.input_image,
            name='lowBand_E',
            is_training=self.is_training,
            enable_bn=self.enable_bn
        )

        # generator: z + label --> generated image
        self.G = self.generator(
            z=self.z,
            y=self.age,
            gender=self.gender,
            enable_tile_label=self.enable_tile_label,
            tile_ratio=self.tile_ratio,
            name='lowBand_G',
            is_training=self.is_training,
            enable_bn=self.enable_bn
        )
        self.G_res = self.generator(
            z=self.z,  # self.z_res
            y=self.age,
            gender=self.gender,
            enable_tile_label=self.enable_tile_label,
            tile_ratio=self.tile_ratio,
            name='highBand_G',
            enable_bn=True,
            is_training=self.is_training
        )

        # discriminator on z
        self.D_z, self.D_z_logits = self.discriminator_z(
            z=self.z,
            is_training=self.is_training
        )

        # discriminator on G
        self.D_G, self.D_G_logits = self.discriminator_img(
            image=self.G + self.G_res,
            y=self.age,
            gender=self.gender,
            is_training=self.is_training,
            enable_bn=True
        )

        # discriminator on z_prior
        self.D_z_prior, self.D_z_prior_logits = self.discriminator_z(
            z=self.z_prior,
            is_training=self.is_training,
            reuse_variables=True
        )

        # discriminator on input image
        self.D_input, self.D_input_logits = self.discriminator_img(
            image=self.input_image,
            y=self.age,
            gender=self.gender,
            is_training=self.is_training,
            reuse_variables=True,
            enable_bn=True
        )

        # ************************************* loss functions *******************************************************
        # loss function of encoder + generator
        #self.EG_loss = tf.nn.l2_loss(self.input_image - self.G) / self.size_batch  # L2 loss
        self.EG_loss = tf.reduce_mean(tf.abs(self.input_image - self.G))  # L1 loss
        self.res_loss = tf.reduce_mean(tf.abs(self.input_image - (self.G + self.G_res * self.params[0])))

        # loss function of discriminator on z
        self.D_z_loss_prior = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_z_prior_logits, tf.ones_like(self.D_z_prior_logits))
        )
        self.D_z_loss_z = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_z_logits, tf.zeros_like(self.D_z_logits))
        )
        self.E_z_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_z_logits, tf.ones_like(self.D_z_logits))
        )
        # self.D_z_loss_prior = .5 * tf.reduce_mean((self.D_z_prior_logits - 1) ** 2)
        # self.D_z_loss_z = .5 * tf.reduce_mean(self.D_z_logits ** 2)
        # self.E_z_loss = .5 * tf.reduce_mean((self.D_z_logits - 1) ** 2)

        # loss function of discriminator on image
        self.D_img_loss_input = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_input_logits, tf.ones_like(self.D_input_logits))
        )
        self.D_img_loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_G_logits, tf.zeros_like(self.D_G_logits))
        )
        self.G_img_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_G_logits, tf.ones_like(self.D_G_logits))
        )
        # self.D_img_loss_input = .5 * tf.reduce_mean((self.D_input_logits - 1) ** 2)
        # self.D_img_loss_G = .5 * tf.reduce_mean(self.D_G_logits ** 2)
        # self.G_img_loss = .5 * tf.reduce_mean((self.D_G_logits - 1) ** 2)

        # *********************************** trainable variables ****************************************************
        trainable_variables = tf.trainable_variables()
        # variables of encoder
        self.E_variables = [var for var in trainable_variables if 'lowBand_E' in var.name]
        # variables of generator
        self.G_variables = [var for var in trainable_variables if 'lowBand_G' in var.name]
        self.G_res_variables = [var for var in trainable_variables if 'highBand_G' in var.name]
        # variables of discriminator on z
        self.D_z_variables = [var for var in trainable_variables if 'D_z_' in var.name]
        # variables of discriminator on image
        self.D_img_variables = [var for var in trainable_variables if 'D_img_' in var.name]

        # ************************************* collect the summary ***************************************
        self.z_summary = tf.summary.histogram('z', self.z)
        self.z_prior_summary = tf.summary.histogram('z_prior', self.z_prior)
        self.EG_loss_summary = tf.summary.scalar('EG_loss', self.EG_loss)
        self.res_loss_summary = tf.summary.scalar('res_loss', self.res_loss)
        self.D_z_loss_z_summary = tf.summary.scalar('D_z_loss_z', self.D_z_loss_z)
        self.D_z_loss_prior_summary = tf.summary.scalar('D_z_loss_prior', self.D_z_loss_prior)
        self.E_z_loss_summary = tf.summary.scalar('E_z_loss', self.E_z_loss)
        self.D_z_logits_summary = tf.summary.histogram('D_z_logits', self.D_z_logits)
        self.D_z_prior_logits_summary = tf.summary.histogram('D_z_prior_logits', self.D_z_prior_logits)
        self.D_img_loss_input_summary = tf.summary.scalar('D_img_loss_input', self.D_img_loss_input)
        self.D_img_loss_G_summary = tf.summary.scalar('D_img_loss_G', self.D_img_loss_G)
        self.G_img_loss_summary = tf.summary.scalar('G_img_loss', self.G_img_loss)
        self.D_G_logits_summary = tf.summary.histogram('D_G_logits', self.D_G_logits)
        self.D_input_logits_summary = tf.summary.histogram('D_input_logits', self.D_input_logits)
        # for saving the graph and variables
        self.saver = tf.train.Saver(max_to_keep=2)

    def train(self,
              num_epochs=200,  # number of epochs
              learning_rate=0.0002,  # learning rate of optimizer
              beta1=0.5,  # parameter for Adam optimizer
              decay_rate=1.0,  # learning rate decay (0, 1], 1 means no decay
              enable_shuffle=True,  # enable shuffle of the dataset
              use_trained_model=True,  # used the saved checkpoint to initialize the model
              params=(1.0, 1.0)  # supposed to be positive float, other values denote auto setting
              ):

        # *************************** load file names of images ******************************************************
        file_names = glob(os.path.join('./data', self.dataset_name, '*.jpg'))
        size_data = len(file_names)
        np.random.seed(seed=1234)
        if enable_shuffle:
            np.random.shuffle(file_names)
        file_names = file_names[:5000]
        # *********************************** optimizer **************************************************************
        # over all, there are three loss functions, weights may differ from the paper because of different datasets
        self.loss_EG = self.EG_loss
        self.loss_res = self.G_img_loss * self.params[0]
        self.loss_Dz = self.D_z_loss_prior + self.D_z_loss_z
        self.loss_Di = self.D_img_loss_input + self.D_img_loss_G

        # set learning rate decay
        self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')
        EG_learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=self.EG_global_step,
            decay_steps=size_data / self.size_batch * 2,
            decay_rate=decay_rate,
            staircase=True
        )

        # optimizer for encoder + generator
        self.EG_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_EG,
            global_step=self.EG_global_step,
            var_list=self.E_variables + self.G_variables
        )
        self.res_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_res,
            global_step=self.EG_global_step,
            var_list=self.G_res_variables
        )

        # optimizer for discriminator on z
        self.D_z_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_Dz,
            var_list=self.D_z_variables
        )
        self.E_z_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.E_z_loss,
            var_list=self.E_variables
        )

        # optimizer for discriminator on image
        self.D_img_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_Di,
            var_list=self.D_img_variables
        )

        # *********************************** tensorboard *************************************************************
        # for visualization (TensorBoard): $ tensorboard --logdir path/to/log-directory
        self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        self.param0_summary = tf.summary.scalar('param0', self.params[0])
        self.param1_summary = tf.summary.scalar('param1', self.params[1])
        self.summary = tf.summary.merge([
            self.z_summary, self.z_prior_summary,
            self.D_z_loss_z_summary, self.D_z_loss_prior_summary,
            self.D_z_logits_summary, self.D_z_prior_logits_summary,
            self.EG_loss_summary, self.E_z_loss_summary, self.res_loss_summary,
            self.D_img_loss_input_summary, self.D_img_loss_G_summary,
            self.G_img_loss_summary, self.EG_learning_rate_summary,
            self.D_G_logits_summary, self.D_input_logits_summary,
            self.param0_summary, self.param1_summary
        ])
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)

        # ************* get some random samples as testing data to visualize the learning process *********************
        sample_files = file_names[0:self.size_batch]
        # file_names[0:self.size_batch] = []
        sample = [load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
        ) for sample_file in sample_files]
        if self.num_input_channels == 1:
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        sample_label_age = np.ones(
            shape=(len(sample_files), self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]
        sample_label_gender = np.ones(
            shape=(len(sample_files), 2),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i, label in enumerate(sample_files):
            label = int(str(sample_files[i]).split('/')[-1].split('_')[0])
            if 0 <= label <= 5:
                label = 0
            elif 6 <= label <= 10:
                label = 1
            elif 11 <= label <= 15:
                label = 2
            elif 16 <= label <= 20:
                label = 3
            elif 21 <= label <= 30:
                label = 4
            elif 31 <= label <= 40:
                label = 5
            elif 41 <= label <= 50:
                label = 6
            elif 51 <= label <= 60:
                label = 7
            elif 61 <= label <= 70:
                label = 8
            else:
                label = 9
            sample_label_age[i, label] = self.image_value_range[-1]
            gender = int(str(sample_files[i]).split('/')[-1].split('_')[1])
            sample_label_gender[i, gender] = self.image_value_range[-1]

        # ******************************************* training *******************************************************
        print '\n\tPreparing for training ...'

        # initialize the graph
        tf.global_variables_initializer().run()

        # load check point
        if use_trained_model:
            if self.load_checkpoint():
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")

        # epoch iteration
        num_batches = len(file_names) // self.size_batch

        # params setting
        flag_auto_setting = [False, False]
        for i, p in enumerate(params):
            try:
                assert type(p) is float or int
                assert p >= 0
            except:
                params[i] = 0
                flag_auto_setting[i] = True

        for epoch in range(num_epochs):
            if enable_shuffle:
                np.random.shuffle(file_names)
            for ind_batch in range(num_batches):
                start_time = time.time()
                # read batch images and labels
                batch_files = file_names[ind_batch*self.size_batch:(ind_batch+1)*self.size_batch]
                batch = [load_image(
                    image_path=batch_file,
                    image_size=self.size_image,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                ) for batch_file in batch_files]
                if self.num_input_channels == 1:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)
                batch_label_age = np.ones(
                    shape=(len(batch_files), self.num_categories),
                    dtype=np.float
                ) * self.image_value_range[0]
                batch_label_gender = np.ones(
                    shape=(len(batch_files), 2),
                    dtype=np.float
                ) * self.image_value_range[0]
                for i, label in enumerate(batch_files):
                    label = int(str(batch_files[i]).split('/')[-1].split('_')[0])
                    if 0 <= label <= 5:
                        label = 0
                    elif 6 <= label <= 10:
                        label = 1
                    elif 11 <= label <= 15:
                        label = 2
                    elif 16 <= label <= 20:
                        label = 3
                    elif 21 <= label <= 30:
                        label = 4
                    elif 31 <= label <= 40:
                        label = 5
                    elif 41 <= label <= 50:
                        label = 6
                    elif 51 <= label <= 60:
                        label = 7
                    elif 61 <= label <= 70:
                        label = 8
                    else:
                        label = 9
                    batch_label_age[i, label] = self.image_value_range[-1]
                    gender = int(str(batch_files[i]).split('/')[-1].split('_')[1])
                    batch_label_gender[i, gender] = self.image_value_range[-1]

                # prior distribution on the prior of z
                batch_z_prior = np.random.uniform(
                    self.image_value_range[0],
                    self.image_value_range[-1],
                    [self.size_batch, self.num_z_channels]
                ).astype(np.float32)

                # update
                _, _, _, _, EG_err, res_err, Ez_err, Dz_err, Dzp_err, Gi_err, DiG_err, Di_err = self.session.run(
                    fetches=[
                        self.EG_optimizer,
                        self.res_optimizer,
                        self.D_z_optimizer,
                        self.D_img_optimizer,
                        # self.E_z_optimizer,
                        self.EG_loss,
                        self.res_loss,
                        self.E_z_loss,
                        self.D_z_loss_z,
                        self.D_z_loss_prior,
                        self.G_img_loss,
                        self.D_img_loss_G,
                        self.D_img_loss_input
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.age: batch_label_age,
                        self.gender: batch_label_gender,
                        self.z_prior: batch_z_prior,
                        self.params: params
                    }
                )
                # update residual twice
                _ = self.session.run(
                    fetches=[
                        self.res_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.age: batch_label_age,
                        self.gender: batch_label_gender,
                        self.z_prior: batch_z_prior,
                        self.params: params
                    }
                )

                print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tEG_err=%.4f\tres_err=%.4f" %
                    (epoch+1, num_epochs, ind_batch+1, num_batches, EG_err, res_err))
                # print("\tEz=%.4f\tDz=%.4f\tDzp=%.4f" % (Ez_err, Dz_err, Dzp_err))
                print("\tGi=%.4f\tDi=%.4f\tDiG=%.4f" % (Gi_err, Di_err, DiG_err))

                # estimate left run time
                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * num_batches + (num_batches - ind_batch - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

                # add to summary
                summary = self.summary.eval(
                    feed_dict={
                        self.input_image: batch_images,
                        self.age: batch_label_age,
                        self.gender: batch_label_gender,
                        self.z_prior: batch_z_prior,
                        self.params: params
                    }
                )
                self.writer.add_summary(summary, self.EG_global_step.eval())

                # update the params
                if flag_auto_setting[0]:
                    params[0] = 1e-4#params[0] -= .1 * (res_err - EG_err)
                    #params[0] = np.clip(params[0], 0, 1)
                if flag_auto_setting[1]:
                    params[1] -= .1 * (res_err - EG_err)
                    params[1] = np.clip(params[1], 0, 1)

            # save sample images for each epoch
            name = '{:02d}'.format(epoch+1)
            self.sample(sample_images, sample_label_age, sample_label_gender, name)
            self.test(sample_images, sample_label_gender, name)

            # save checkpoint for each 10 epoch
            if np.mod(epoch, 10) == 9:
                self.save_checkpoint()

        # save the trained model
        self.save_checkpoint()
        # close the summary writer
        self.writer.close()

    def encoder(self, image, reuse_variables=False, name='encoder', enable_bn=False, is_training=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            current = conv2d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.size_kernel,
                    name=name + '_conv' + str(i)
                )
            if enable_bn:
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=True,
                    is_training=is_training,
                    scope=name + '_bn' + str(i),
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)

        # fully connection layer
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=self.num_z_channels,
            name=name + '_fc'
        )
        if enable_bn:
            current = tf.contrib.layers.batch_norm(
                current,
                scale=False,
                is_training=is_training,
                scope=name + '_bn' + '_fc',
                reuse=reuse_variables
            )

        # output
        return tf.nn.tanh(current)

    def generator(self, z, y, gender, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0, name='decoder',
                  enable_bn=False, is_training=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / self.num_categories)
        else:
            duplicate = 1
        z = concat_label(z, y, duplicate=duplicate)
        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / 2)
        else:
            duplicate = 1
        z = concat_label(z, gender, duplicate=duplicate)
        size_mini_map = int(self.size_image / 2 ** num_layers)
        # fc layer
        current = fc(
            input_vector=z,
            num_output_length=self.num_gen_channels * size_mini_map * size_mini_map,
            name=name + '_fc'
        )
        if enable_bn:
            current = tf.contrib.layers.batch_norm(
                current,
                scale=False,
                is_training=is_training,
                scope=name + '_bn' + '_fc',
                reuse=reuse_variables
            )
        # reshape to cube for deconv
        current = tf.reshape(current, [-1, size_mini_map, size_mini_map, self.num_gen_channels])
        current = tf.nn.relu(current)
        # deconv layers with stride 2
        for i in range(num_layers):
            current = deconv2d(
                    input_map=current,
                    output_shape=[self.size_batch,
                                  size_mini_map * 2 ** (i + 1),
                                  size_mini_map * 2 ** (i + 1),
                                  int(self.num_gen_channels / 2 ** (i + 1))],
                    size_kernel=self.size_kernel,
                    name=name + '_deconv' + str(i)
                )
            if enable_bn:
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name + '_bn' + str(i),
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)
        current = deconv2d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.size_image,
                          self.size_image,
                          int(self.num_gen_channels / 2 ** (i + 2))],
            size_kernel=self.size_kernel,
            stride=1,
            name=name + '_deconv' + str(i+1)
        )
        if enable_bn:
            current = tf.contrib.layers.batch_norm(
                current,
                scale=False,
                is_training=is_training,
                scope=name + '_bn' + str(i+1),
                reuse=reuse_variables
            )
        current = tf.nn.relu(current)
        current = deconv2d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.size_image,
                          self.size_image,
                          self.num_input_channels],
            size_kernel=self.size_kernel,
            stride=1,
            name=name + '_deconv' + str(i + 2)
        )

        # output
        return tf.nn.tanh(current)

    def discriminator_z(self, z, is_training=True, reuse_variables=False, num_hidden_layer_channels=(64, 32, 16), enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        current = z
        # fully connection layer
        for i in range(len(num_hidden_layer_channels)):
            name = 'D_z_fc' + str(i)
            current = fc(
                    input_vector=current,
                    num_output_length=num_hidden_layer_channels[i],
                    name=name
                )
            if enable_bn:
                name = 'D_z_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)
        # output layer
        name = 'D_z_fc' + str(i+1)
        current = fc(
            input_vector=current,
            num_output_length=1,
            name=name
        )
        return tf.nn.sigmoid(current), current

    def discriminator_img(self, image, y, gender, is_training=True, reuse_variables=False, enable_bn=True,
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
            if i == 0:
                current = concat_label(current, y)
                current = concat_label(current, gender, int(self.num_categories / 2))
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

    def unet(self, image, y=None, gender=None, name='unet'):
        s = self.size_image
        self.gf_dim = self.num_encoder_channels
        s2, s4, s8, s16, s32, s64 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64)

        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
        # e1 is (128 x 128 x self.gf_dim)
        e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name=name + 'g_e2_conv'))
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name=name + 'g_e3_conv'))
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name=name + 'g_e4_conv'))
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name=name + 'g_e5_conv'))
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name=name + 'g_e6_conv'))
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name=name + 'g_e7_conv'))
        # e7 is (2 x 2 x self.gf_dim*8)
        # e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name=name + 'g_e8_conv'))
        # e8 is (1 x 1 x self.gf_dim*8)

        self.d1 = deconv2d(tf.nn.relu(e7), [self.size_batch, s64, s64, self.gf_dim*8], name=name + 'g_d1')
        d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
        d1 = tf.concat(3, [d1, e6])
        # d1 is (2 x 2 x self.gf_dim*8*2)

        self.d2 = deconv2d(tf.nn.relu(d1), [self.size_batch, s32, s32, self.gf_dim*8], name=name + 'g_d2')
        d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
        d2 = tf.concat(3, [d2, e5])
        # d2 is (4 x 4 x self.gf_dim*8*2)

        self.d3 = deconv2d(tf.nn.relu(d2), [self.size_batch, s16, s16, self.gf_dim*8], name=name + 'g_d3')
        d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
        d3 = tf.concat(3, [d3, e4])
        # d3 is (8 x 8 x self.gf_dim*8*2)

        self.d4 = deconv2d(tf.nn.relu(d3), [self.size_batch, s8, s8, self.gf_dim*4], name=name + 'g_d4')
        d4 = self.g_bn_d4(self.d4)
        d4 = tf.concat(3, [d4, e3])
        # d4 is (16 x 16 x self.gf_dim*8*2)

        self.d5 = deconv2d(tf.nn.relu(d4), [self.size_batch, s4, s4, self.gf_dim*2], name=name + 'g_d5')
        d5 = self.g_bn_d5(self.d5)
        d5 = tf.concat(3, [d5, e2])
        # d5 is (32 x 32 x self.gf_dim*4*2)

        self.d6 = deconv2d(tf.nn.relu(d5), [self.size_batch, s2, s2, self.gf_dim], name=name + 'g_d6')
        d6 = self.g_bn_d6(self.d6)
        d6 = tf.concat(3, [d6, e1])
        # d6 is (64 x 64 x self.gf_dim*2*2)

        # self.d7 = deconv2d(tf.nn.relu(d6), [self.size_batch, s2, s2, self.gf_dim], name=name + 'g_d7')
        # d7 = self.g_bn_d7(self.d7)
        # d7 = tf.concat(3, [d7, e1])
        # d7 is (128 x 128 x self.gf_dim*1*2)

        self.d7 = deconv2d(tf.nn.relu(d6), [self.size_batch, s, s, self.num_input_channels], name=name + 'g_d7')
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(self.d7)

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, 'model'),
            global_step=self.EG_global_step.eval()
        )

    def load_checkpoint(self):
        print("\n\tLoading pre-trained model ...")
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
            return True
        else:
            return False

    def sample(self, images, labels, gender, name):
        sample_dir = os.path.join(self.save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        G, G_res = self.session.run(
            [self.G, self.G_res],
            feed_dict={
                self.input_image: images,
                self.age: labels,
                self.gender: gender
            }
        )
        size_frame = int(np.sqrt(self.size_batch))
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(sample_dir, name + '_G.png'),
            image_value_range=self.image_value_range,
            size_frame=[size_frame, size_frame]
        )
        save_batch_images(
            batch_images=G_res,
            save_path=os.path.join(sample_dir, name + '_res.png'),
            image_value_range=self.image_value_range,
            size_frame=[size_frame, size_frame]
        )
        save_batch_images(
            batch_images=G + G_res,
            save_path=os.path.join(sample_dir, name + '_Gres.png'),
            image_value_range=self.image_value_range,
            size_frame=[size_frame, size_frame]
        )

    def test(self, images, gender, name):
        test_dir = os.path.join(self.save_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        images = images[:int(np.sqrt(self.size_batch)), :, :, :]
        gender = gender[:int(np.sqrt(self.size_batch)), :]
        size_sample = images.shape[0]
        labels = np.arange(size_sample)
        labels = np.repeat(labels, size_sample)
        query_labels = np.ones(
            shape=(size_sample ** 2, self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i in range(query_labels.shape[0]):
            query_labels[i, labels[i]] = self.image_value_range[-1]
        query_images = np.tile(images, [size_sample, 1, 1, 1])
        query_gender = np.tile(gender, [size_sample, 1])
        G, G_res = self.session.run(
            [self.G, self.G_res],
            feed_dict={
                self.input_image: query_images,
                self.age: query_labels,
                self.gender: query_gender
            }
        )
        save_batch_images(
            batch_images=query_images,
            save_path=os.path.join(test_dir, 'input.png'),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(test_dir, name + '_G.png'),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )
        save_batch_images(
            batch_images=G_res,
            save_path=os.path.join(test_dir, name + '_res.png'),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )
        save_batch_images(
            batch_images=G + G_res,
            save_path=os.path.join(test_dir, name + '_Gres.png'),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )

    def custom_test(self, testing_samples_dir):
        if not self.load_checkpoint():
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")

        num_samples = int(np.sqrt(self.size_batch))
        file_names = glob(testing_samples_dir)
        if len(file_names) < num_samples:
            print 'The number of testing images is must larger than %d' % num_samples
            exit(0)
        sample_files = file_names[0:num_samples]
        sample = [load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
        ) for sample_file in sample_files]
        if self.num_input_channels == 1:
            images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            images = np.array(sample).astype(np.float32)
        gender_male = np.ones(
            shape=(num_samples, 2),
            dtype=np.float32
        ) * self.image_value_range[0]
        gender_female = np.ones(
            shape=(num_samples, 2),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i in range(gender_male.shape[0]):
            gender_male[i, 0] = self.image_value_range[-1]
            gender_female[i, 1] = self.image_value_range[-1]

        self.test(images, gender_male, 'test_as_male.png')
        self.test(images, gender_female, 'test_as_female.png')

        print '\n\tDone! Results are saved as %s\n' % os.path.join(self.save_dir, 'test', 'test_as_xxx.png')


