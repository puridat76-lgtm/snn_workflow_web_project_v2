from pathlib import Path

import tensorflow as tf
from tensorflow import keras


OUT = Path(__file__).resolve().parent
IMAGE_SHAPE = (224, 224, 3)


def build_encoder():
  image = keras.Input(shape=IMAGE_SHAPE, name="cat_face")
  x = keras.layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenetv2_preprocess")(image)
  base = keras.applications.MobileNetV2(
    input_shape=IMAGE_SHAPE,
    include_top=False,
    pooling="avg",
    weights="imagenet",
  )
  base.trainable = False
  embedding = base(x)
  embedding = keras.layers.UnitNormalization(axis=-1, name="l2_embedding")(embedding)
  return keras.Model(image, embedding, name="mobilenet_embedding_encoder")


def build_pair_model(encoder):
  left = keras.Input(shape=IMAGE_SHAPE, name="left_cat_face")
  right = keras.Input(shape=IMAGE_SHAPE, name="right_cat_face")
  left_embedding = encoder(left)
  right_embedding = encoder(right)
  cosine = keras.layers.Dot(axes=1, normalize=False, name="cosine_similarity")([
    left_embedding,
    right_embedding,
  ])
  score = keras.layers.Rescaling(scale=0.5, offset=0.5, name="similarity_score")(cosine)
  return keras.Model([left, right], score, name="mobilenet_embedding_cosine_pair")


def main():
  encoder = build_encoder()
  pair_model = build_pair_model(encoder)

  encoder.save(OUT / "mobilenet_embedding_encoder.keras")
  pair_model.save(OUT / "mobilenet_pair_similarity.keras")

  encoder.export(OUT / "mobilenet_embedding_encoder_savedmodel")
  pair_model.export(OUT / "mobilenet_pair_similarity_savedmodel")

  print("Saved:")
  print(OUT / "mobilenet_embedding_encoder.keras")
  print(OUT / "mobilenet_pair_similarity.keras")


if __name__ == "__main__":
  main()
