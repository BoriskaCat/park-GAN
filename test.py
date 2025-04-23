import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import cv2


parser = argparse.ArgumentParser(description="Generate masterplan from contour image and area.")
parser.add_argument("--img", type=str, required=True, help="Path to the contour image.")
parser.add_argument("--area", type=float, required=True, help="Area value in hectares.")
args = parser.parse_args()

IMG_HEIGHT = 512
IMG_WIDTH = 512
EMBED_DIM = 8
MODELS_DIR = "./nets"


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    block = tf.keras.Sequential()
    block.add(layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())
    return block

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    block = tf.keras.Sequential()
    block.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    block.add(layers.BatchNormalization())
    if apply_dropout:
        block.add(layers.Dropout(0.5))
    block.add(layers.ReLU())
    return block

def build_generator(in_channels=3+EMBED_DIM, out_channels=3):
    inputs = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, in_channels])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4),
        upsample(512, 4),
        upsample(512, 4),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(out_channels, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

class AreaEmbedder(tf.keras.Model):
    def __init__(self, embedding_dim=8):
        super().__init__()
        self.d1 = layers.Dense(16, activation='relu')
        self.d2 = layers.Dense(embedding_dim)

    def call(self, area_tensor, training=False):
        x = self.d1(area_tensor, training=training)
        x = self.d2(x, training=training)
        return x

@tf.function
def combine_with_area(img, area, embedder):
    area_2d = tf.expand_dims(area, axis=-1)
    area_emb = embedder(area_2d, training=False)
    area_emb_tiled = tf.tile(tf.reshape(area_emb, [tf.shape(area_emb)[0], 1, 1, EMBED_DIM]),
                             [1, IMG_HEIGHT, IMG_WIDTH, 1])
    return tf.concat([img, area_emb_tiled], axis=-1)

generator = build_generator()
area_embedder = AreaEmbedder(embedding_dim=EMBED_DIM)
area_embedder.build(input_shape=(None, 1))

generator.load_weights(os.path.join(MODELS_DIR, "generator_epoch_1500.weights.h5"))
area_embedder.load_weights(os.path.join(MODELS_DIR, "area_embedder_epoch_1500.weights.h5"))
area_embedder.trainable = False
generator.trainable = False
print("Models loaded.")

img = Image.open(args.img).convert('RGB')
original_width, original_height = img.size
img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
img_array = np.array(img_resized, dtype=np.float32) / 255.0
img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

area = tf.constant([args.area], dtype=tf.float32)
contour_with_area = combine_with_area(img_array, area, area_embedder)

fake_masterplan = generator(contour_with_area, training=False)
fake_masterplan = (fake_masterplan + 1) / 2.0  # Denormalize to [0,1]
fake_masterplan_img = fake_masterplan[0].numpy()
fake_masterplan_img = (fake_masterplan_img * 255).astype(np.uint8)

fake_masterplan_resized = cv2.resize(fake_masterplan_img, (original_width, original_height), interpolation=cv2.INTER_AREA)

output_path = os.path.splitext(args.img)[0] + f"_generated.jpg"
fake_masterplan_pil = Image.fromarray(fake_masterplan_resized)
fake_masterplan_pil.save(output_path)
print(f"Output saved to {output_path}")