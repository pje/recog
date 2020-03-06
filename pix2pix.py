# pix2pix using tensorflow2, based on https://www.tensorflow.org/tutorials/generative/pix2pix

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
from glob import glob
import os
from pathlib import Path
import tarfile
import time
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

DATASET_NAME = 'flickr_clouds'
IMG_SIZE = 256 # images must be square

ROOT_DIR = Path().resolve()
UNIQUE_SESSION_NAME = DATASET_NAME + '_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

# import ipdb; ipdb.set_trace()

try:
    from google.colab import drive
except ImportError: # we're NOT running on colab. use local filesystem
    CHECKPOINTS_DIR = os.path.join(ROOT_DIR, 'checkpoints')
    LOGS_DIR = os.path.join(ROOT_DIR, 'training_logs')
    DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
    DATASET_DIR = os.path.join(DATASETS_DIR, DATASET_NAME)
else: # we ARE running on colab. Use Drive for file reads/writes
    drive.mount('/content/gdrive')
    DRIVE_ROOT = os.path.join(ROOT_DIR, 'gdrive', 'My Drive')
    LOGS_DIR = os.path.join(DRIVE_ROOT, 'training_logs')
    CHECKPOINTS_DIR = os.path.join(DRIVE_ROOT, 'checkpoints')
    DATASETS_DIR = os.path.join(DRIVE_ROOT, 'datasets')
    DATASET_DIR = os.path.join(DATASETS_DIR, DATASET_NAME)
    if not os.path.isdir(os.path.join(DATASET_DIR)):
        tar = tarfile.open(os.path.join(DATASETS_DIR, DATASET_NAME+'.tar.gz'))
        tar.extractall(path=DATASETS_DIR)
        tar.close()

CHECKPOINT_DIR = os.path.join(CHECKPOINTS_DIR, DATASET_NAME)
TFJS_EXPORT_DIR = os.path.join(ROOT_DIR, 'browser', 'tensorflow_js_models', DATASET_NAME)
CHECKPOINT_PREFIX = 'ckpt'
LOG_DIR = os.path.join(LOGS_DIR, UNIQUE_SESSION_NAME)
BUFFER_SIZE = 400
BATCH_SIZE = 1
OUTPUT_CHANNELS = 3
LAMBDA = 100
MAX_EPOCHS = 140
CHECKPOINT_SAVE_FREQUENCY = 20 # in epochs

SUMMARY_WRITER = tf.summary.create_file_writer(LOG_DIR)

def get_methods(obj, spacing=20):
  methodList = []
  for method_name in dir(obj):
    try:
        if callable(getattr(obj, method_name)):
            methodList.append(str(method_name))
    except:
        methodList.append(str(method_name))
  processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
  for method in methodList:
    try:
        print(str(method.ljust(spacing)) + ' ' +
              processFunc(str(getattr(obj, method).__doc__)[0:90]))
    except:
        print(method.ljust(spacing) + ' ' + ' getattr() failed')


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    # remove alpha channel if present
    image = tf.cond(tf.equal(tf.shape(image)[2], 4), lambda: image[:,:,:3], lambda: image)
    # convert grayscale to RGB
    image = tf.cond(tf.equal(tf.shape(image)[2], 1), lambda: tf.image.grayscale_to_rgb(image), lambda: image)
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
        size=[2, IMG_SIZE, IMG_SIZE, 3]
    )
    return cropped_image[0], cropped_image[1]


# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    scaling_factor = 1.1171875 # turns 256 into 286 (as the original paper used)
    scaled_size = round(scaling_factor * IMG_SIZE)
    # resize to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, scaled_size, scaled_size)
    # ...then randomly crop it back down to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image


def downsample(filters, kernel_size, name, apply_batchnorm=True, **kwargs):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False,
            **kwargs
        )
    )
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, kernel_size, name, apply_dropout=False, **kwargs):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False,
            **kwargs
        )
    )
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3], name="generator_input")
    down_stack = [
        downsample(64,  4, name="downsample_1", apply_batchnorm=False, input_shape=(256, 256, 3)),
        downsample(128, 4, name="downsample_2", apply_batchnorm=True,  input_shape=(128, 128, 64)),
        downsample(256, 4, name="downsample_3", apply_batchnorm=True,  input_shape=(64, 64, 128)),
        downsample(512, 4, name="downsample_4", apply_batchnorm=True,  input_shape=(32, 32, 256)),
        downsample(512, 4, name="downsample_5", apply_batchnorm=True,  input_shape=(16, 16, 512)),
        downsample(512, 4, name="downsample_6", apply_batchnorm=True,  input_shape=(8, 8, 512)),
        downsample(512, 4, name="downsample_7", apply_batchnorm=True,  input_shape=(4, 4, 512)),
        downsample(512, 4, name="downsample_8", apply_batchnorm=True,  input_shape=(2, 2, 512)),
    ]

    up_stack = [
        upsample(512, 4, name="upsample_1", apply_dropout=True,  input_shape=(1, 1, 512)),
        upsample(512, 4, name="upsample_2", apply_dropout=True,  input_shape=(2, 2, 1024)),
        upsample(512, 4, name="upsample_3", apply_dropout=True,  input_shape=(4, 4, 1024)),
        upsample(512, 4, name="upsample_4", apply_dropout=False, input_shape=(8, 8, 1024)),
        upsample(256, 4, name="upsample_5", apply_dropout=False, input_shape=(16, 16, 1024)),
        upsample(128, 4, name="upsample_6", apply_dropout=False, input_shape=(32, 32, 512)),
        upsample(64,  4, name="upsample_7", apply_dropout=False, input_shape=(64, 64, 256)),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS, 4,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        activation='tanh',
        name="generator_output"
    )
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
    inp = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    down1 = downsample(round(IMG_SIZE / 4), 4, name="down1", apply_batchnorm=False)(x)
    down2 = downsample(round(IMG_SIZE / 2), 4, name="down2",)(down1)
    down3 = downsample(round(IMG_SIZE * 1), 4, name="down3",)(down2)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(
        round(IMG_SIZE * 2),
        4,
        strides=1,
        kernel_initializer=initializer,
        use_bias=False
    )(zero_pad1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(
        1,
        4,
        strides=1,
        kernel_initializer=initializer
    )(zero_pad2)
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


@tf.function()
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


def fit(generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint, train_dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        print("Epoch ", epoch)

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

        # generate & save a random image at the end of every epoch
        for example_input, _example_target in train_dataset.take(1):
            prediction = generator(example_input, training=True)
            encoded_image = tf.image.encode_jpeg(tf.dtypes.cast((prediction[0] * 0.5 + 0.5) * 255, tf.uint8))
            tf.io.write_file(
                os.path.join(
                    LOG_DIR,
                    UNIQUE_SESSION_NAME + "_epoch_" + str(epoch).zfill(4) + ".jpg"
                ),
                encoded_image
            )

        print(
            'Epoch {} took {} sec\n'.format(epoch, time.time() - start)
        )

        # save our checkpoint every 20 epochs (this is slow as model size grows)
        if (epoch != 0) and (epoch % CHECKPOINT_SAVE_FREQUENCY == 0):
            print('Saving checkpoint to {}\n'.format(CHECKPOINT_DIR))
            checkpoint.write(os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX))

    checkpoint.write(os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX))


def main(operation):
    def get_image_files(directory):
        if not os.path.isdir(directory):
            raise Exception("Not a directory: {}".format(directory))
        else:
            train_files = glob(os.path.join(directory, '**', '*.png'), recursive=True) + glob(os.path.join(directory, '**', '*.jpg'), recursive=True)
            if len(train_files) < 1:
                raise Exception("No training images exist in {}".format(directory))
            return train_files


    def dataset_from_images(image_files):
        train_dataset = tf.data.Dataset.list_files(image_files)
        train_dataset = train_dataset.map(
            load_image_train,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        train_dataset = train_dataset.shuffle(BUFFER_SIZE)
        train_dataset = train_dataset.batch(BATCH_SIZE)
        return train_dataset


    if operation == 'train' or operation == 'export':
        train_dataset = dataset_from_images(get_image_files(DATASET_DIR))
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
        # a little unusual because we only store 1 checkpoint (storage constraints)
        latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if latest_checkpoint:
            status = checkpoint.restore(latest_checkpoint)
            status.assert_existing_objects_matched()
            print("Restored from {}".format(latest_checkpoint))
        else:
            print(
                "No checkpoint found in {}. Initializing from scratch.".format(
                    CHECKPOINT_DIR
                )
            )

        if operation == 'train':
            fit(
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                checkpoint=checkpoint,
                train_dataset=train_dataset,
                epochs=MAX_EPOCHS
            )
        elif operation == 'export':
            # we have to do a phony "fit" epoch of a single image so that the optimizer
            # gets initialized properly. without running this, optimizers won't get
            # saved correctly for some reason.
            print('running a single phony training epoch to initialize the optimizers...')
            train_dataset = dataset_from_images(get_image_files(DATASET_DIR))
            fit(
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                checkpoint=checkpoint,
                train_dataset=train_dataset.take(1),
                epochs=1
            )
            save_path = os.path.join('models', DATASET_NAME + '_generator.h5')
            generator.save(save_path) # Keras hd5 format
            print('saved to {}'.format(save_path))

            # converter = tf.lite.TFLiteConverter.from_keras_model(generator)
            # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
            # tflite_quant_model = converter.convert()
            # open(os.path.join('models', DATASET_NAME + ".tflite"), "wb").write(tflite_quant_model)
            # print('saved TFLite quantized model to to {}'.format(os.path.join('models', DATASET_NAME + ".tflite")))

            tfjs.converters.save_keras_model(generator, TFJS_EXPORT_DIR + '_generator') # hd5 -> tfjs (bins & json)
            print('saved to {}'.format(TFJS_EXPORT_DIR + '_generator'))

            # tfjs.converters.convert_tf_saved_model(
            #     os.path.join('models', DATASET_NAME + '_generator'),
            #     output_dir=TFJS_EXPORT_DIR + '_generator',
            # ) # convert "SavedModel" to tfjs

            tf.keras.utils.plot_model(generator, to_file='generator.png', show_shapes=True, show_layer_names=True, expand_nested=True)

    elif operation == 'predict':
        train_dataset = dataset_from_images(get_image_files(DATASET_DIR))
        generator = tf.keras.models.load_model(
            os.path.join(ROOT_DIR, 'models', DATASET_NAME + '_generator.h5')
        )
        i = 0
        for input_image, _example_target in train_dataset.take(100):
            # import ipdb; ipdb.set_trace()
            input_image = tf.image.rot90(input_image)
            prediction = generator(input_image, training=True)[0]
            img_predicted = np.interp(prediction, [-1.0, 1.0], (0.0, 255.0))
            encoded_image = tf.image.encode_jpeg(tf.dtypes.cast(img_predicted, tf.uint8))
            tf.io.write_file(
                os.path.join(
                    LOG_DIR,
                    UNIQUE_SESSION_NAME + "_generated_" + str(i).zfill(4) + ".jpg"
                ),
                encoded_image
            )
            i = i + 1
    else:
        raise Exception("operation must be 'train', 'export', or 'predict'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["train", "export", "predict"])
    a = parser.parse_args()
    main(operation=a.operation)
