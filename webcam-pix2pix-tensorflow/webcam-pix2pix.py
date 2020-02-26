from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import time
import params
import gui
import msa.utils
from msa.capturer import Capturer
from msa.framestats import FrameStats

capture = None # msa.capturer.Capturer, video capture wrapper
generator = None

img_cap = np.empty([]) # captured image before processing
img_in = np.empty([]) # processed capture image
img_out = np.empty([]) # output from prediction model

gui.init_app()

pyqt_params = gui.init_params(params.params_list, target_obj=params, w=320)

# reading & writing to pyqtgraph.parametertree seems to be slow,
# so going to cache in an object for direct access
gui.params_to_obj(pyqt_params, target_obj=params, create_missing=True, verbose=True)

# create main window
gui.init_window(x=320, w=(gui.screen_size().width()-320), h=(gui.screen_size().width()-320)*0.4)

import datetime
from glob import glob
import os
from pathlib import Path
import tarfile
import time
import tensorflow as tf
from matplotlib import pyplot as plt

DATASET_NAME = 'flickr_flowers_AtoB'
ROOT_DIR = '/Users/pje/recog'


try:
    from google.colab import drive
except ImportError: # we're NOT running on colab. use local filesystem
    CHECKPOINTS_DIR = os.path.join(ROOT_DIR, 'checkpoints')
    DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
    DATASET_DIR = os.path.join(DATASETS_DIR, DATASET_NAME)
else: # we ARE running on colab. Use Drive for file reads/writes
    drive.mount('/content/gdrive')
    DRIVE_ROOT = os.path.join(ROOT_DIR, 'gdrive', 'My Drive')
    CHECKPOINTS_DIR = os.path.join(DRIVE_ROOT, 'checkpoints')
    DATASETS_DIR = os.path.join(DRIVE_ROOT, 'datasets')
    DATASET_DIR = os.path.join(DATASETS_DIR, DATASET_NAME)
    if not os.path.isdir(os.path.join(DATASET_DIR)):
        tar = tarfile.open(os.path.join(DATASETS_DIR, DATASET_NAME+'.tar.gz'))
        tar.extractall(path=DATASETS_DIR)
        tar.close()

CHECKPOINT_DIR = os.path.join(CHECKPOINTS_DIR, DATASET_NAME)
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, 'ckpt')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
LOG_DIR = os.path.join(LOGS_DIR, 'fit', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100
EPOCHS = 100
CHECKPOINT_SAVE_FREQUENCY = 20 # in epochs

SUMMARY_WRITER = tf.summary.create_file_writer(LOG_DIR)

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(
        input_image,
        real_image,
        IMG_HEIGHT,
        IMG_WIDTH
    )
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image,
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image,
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image,
        size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )
    return cropped_image[0], cropped_image[1]


# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)
    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
    )
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
    )
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS, 4,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        activation='tanh'
    )  # (bs, 256, 256, 3)
    x = inputs
    # Downsampling through the model
    skips = []

    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_loss(disc_generated_output, gen_output, target):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(
        disc_generated_output),
        disc_generated_output
    )
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)
    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512,
        4,
        strides=1,
        kernel_initializer=initializer,
        use_bias=False
    )(zero_pad1)  # (bs, 31, 31, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
    last = tf.keras.layers.Conv2D(
        1,
        4,
        strides=1,
        kernel_initializer=initializer
    )(zero_pad2)  # (bs, 30, 30, 1)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output),
        disc_generated_output
    )
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generate_images(model, input, target):
    prediction = model(input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [input[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


@tf.function
def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator(
            [input_image, gen_output],
            training=True
        )
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output,
            gen_output,
            target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(
        gen_total_loss,
        generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )
    generator_optimizer.apply_gradients(
        zip(
            generator_gradients,
            generator.trainable_variables
        )
    )
    discriminator_optimizer.apply_gradients(
        zip(
            discriminator_gradients,
            discriminator.trainable_variables
        )
    )
    with SUMMARY_WRITER.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint_manager, train_dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, target) in train_dataset.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            train_step(
                generator=generator,
                discriminator=discriminator,
                discriminator_optimizer=discriminator_optimizer,
                generator_optimizer=generator_optimizer,
                input_image=input_image,
                target=target,
                epoch=epoch
            )
        print()

        if (epoch + 1) % CHECKPOINT_SAVE_FREQUENCY == 0:
            # save our checkpoint!
            checkpoint_manager.save()
            # also: generate a random image from this epoch and save it to disk
            for example_input, _example_target in train_dataset.take(1):
                prediction = generator(example_input, training=True)
                encoded_image = tf.image.encode_jpeg(tf.dtypes.cast((prediction[0] * 0.5 + 0.5) * 255, tf.uint8))
                tf.io.write_file(os.path.join(LOG_DIR, "epoch_" + str(epoch) + ".jpg"), encoded_image)

        print(
            'Time taken for epoch {} is {} sec\n'.format(
                epoch + 1, time.time()-start
            )
        )

    checkpoint_manager.save()
















train_dataset = tf.data.Dataset.list_files(
    glob(os.path.join(DATASET_DIR, '*.jpg')) +
        glob(os.path.join(DATASET_DIR, '*.png'))
)
train_dataset = train_dataset.map(
    load_image_train,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(
    glob(os.path.join(DATASET_DIR, '*.jpg')) +
        glob(os.path.join(DATASET_DIR, '*.png'))
)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

generator = Generator()
discriminator = Discriminator()
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator.compile(
    loss="mean_squared_error",
    optimizer=discriminator_optimizer
)
generator.compile(
    loss="mean_squared_error",
    optimizer=generator_optimizer
)

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint,
    CHECKPOINT_DIR,
    max_to_keep=1
)

if checkpoint_manager.latest_checkpoint:
    status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
    status.assert_existing_objects_matched()
    print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
else:
    print(
        "No existing checkpoint found in CHECKPOINT_DIR, {}!".format(
            CHECKPOINT_DIR
        )
    )
    10 / 0














def init_capture(capture, output_shape):
    if capture:
        capture.close()

    capture_shape = (params.Capture.Init.height, params.Capture.Init.width)
    capture = Capturer(sleep_s = params.Capture.sleep_s,
                       device_id = params.Capture.Init.device_id,
                       capture_shape = capture_shape,
                       capture_fps = params.Capture.Init.fps,
                       output_shape = output_shape
                       )

    capture.update()

    if params.Capture.Init.use_thread:
        capture.start()

    return capture


capture = init_capture(capture, output_shape=[256, 256, 3])
frame_stats = FrameStats('Main')

while not params.Main.quit:

    # reinit capture device if parameters have changed
    if params.Capture.Init.reinitialise:
        params.child('Capture').child('Init').child('reinitialise').setValue(False)
        capture = init_capture(capture, output_shape=[256, 256, 3])


    capture.enabled = params.Capture.enabled
    if params.Capture.enabled:
        # update capture parameters from GUI
        capture.output_shape = [256, 256, 3]
        capture.verbose = params.Main.verbose
        capture.freeze = params.Capture.freeze
        capture.sleep_s = params.Capture.sleep_s
        for p in msa.utils.get_members(params.Capture.Processing):
            setattr(capture, p, getattr(params.Capture.Processing, p))

        # run capture if multithreading is disabled
        if params.Capture.Init.use_thread == False:
            capture.update()

        img_cap = np.copy(capture.img) # create copy to avoid thread issues


    # interpolate (temporal blur) on input image
    img_in = msa.utils.np_lerp( img_in, img_cap, 1 - params.Prediction.pre_time_lerp)

    # run prediction
    if params.Prediction.enabled and generator:
        generator_input = np.expand_dims(img_in, 0) if len(img_in.shape) < 4 else img_in
        img_predicted = generator(generator_input, training=True)[0] * 0.5 + 0.5
        img_predicted = np.interp(img_predicted, [-1.0, 1.0], (0., 1.))
    else:
        img_predicted = capture.img0

    # interpolate (temporal blur) on output image
    img_out = msa.utils.np_lerp(img_out, img_predicted, 1 - params.Prediction.post_time_lerp)

    # update frame states
    frame_stats.verbose = params.Main.verbose
    frame_stats.update()

    # update gui
    gui.update_image(0, capture.img0)
    gui.update_image(1, img_in)
    gui.update_image(2, img_out)
    gui.update_stats(frame_stats.str + "   |   " + capture.frame_stats.str)
    gui.process_events()

    time.sleep(params.Main.sleep_s)

capture.close()
gui.close()

capture = None
generator = None

print('Finished')
