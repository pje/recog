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

DATASET_NAME = 'flickr_flower_photos'
IMG_SIZE = 256 # images must be square

ROOT_DIR = '/Users/pje/recog'
UNIQUE_SESSION_NAME = DATASET_NAME + '_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

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
CHECKPOINT_PREFIX = 'ckpt'
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
LOG_DIR = os.path.join(LOGS_DIR, UNIQUE_SESSION_NAME)
OUTPUT_CHANNELS = 3
LAMBDA = 100

SUMMARY_WRITER = tf.summary.create_file_writer(LOG_DIR)

def console():
    from code import InteractiveConsole
    InteractiveConsole(locals={**globals(), **locals(), **vars()}).interact()


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






generator = tf.keras.models.load_model(os.path.join(ROOT_DIR, 'models', DATASET_NAME + '_generator.h5'))














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


capture = init_capture(capture, output_shape=[IMG_SIZE, IMG_SIZE, 3])
frame_stats = FrameStats('Main')

while not params.Main.quit:

    # reinit capture device if parameters have changed
    if params.Capture.Init.reinitialise:
        params.child('Capture').child('Init').child('reinitialise').setValue(False)
        capture = init_capture(capture, output_shape=[IMG_SIZE, IMG_SIZE, 3])


    capture.enabled = params.Capture.enabled
    if params.Capture.enabled:
        # update capture parameters from GUI
        capture.output_shape = [IMG_SIZE, IMG_SIZE, 3]
        capture.verbose = params.Main.verbose
        capture.freeze = params.Capture.freeze
        capture.sleep_s = params.Capture.sleep_s
        for p in msa.utils.get_members(params.Capture.Processing):
            setattr(capture, p, getattr(params.Capture.Processing, p))

        # run capture if multithreading is disabled
        if params.Capture.Init.use_thread == False:
            capture.update()

        while capture.img is None:
            time.sleep(0.001)

        img_cap = np.copy(capture.img) # create copy to avoid thread issues


    # interpolate (temporal blur) on input image
    img_in = msa.utils.np_lerp( img_in, img_cap, 1 - params.Prediction.pre_time_lerp)

    # run prediction
    if params.Prediction.enabled and generator:
        generator_input = tf.expand_dims(img_in, 0) if len(img_in.shape) < 4 else img_in

        # print('img_in (after expand):\n')
        # tf.print(generator_input)
        # print(generator_input)
        # print("...........\n\n\n\n ")

        generator_input = (generator_input * 2 - 1) # transform values: (0..1.0) -> (-1.0..1.0)
        # print('generator_input:\n')
        # print(generator_input)
        # tf.print(generator_input)
        # print("...........\n\n\n\n ")

        # raise Exception

        img_predicted = generator(generator_input, training=True)[0]
        img_predicted = np.interp(img_predicted, [-1.0, 1.0], (0.0, 1.0))
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
