# model_def.py

import tensorflow as tf
import tensorflow_recommenders as tfrs


# 🧠 مودل التوصية
class RecommendationModel(tfrs.models.Model):

    def __init__(self, user_vocab, item_vocab):
        super().__init__()

        # 👤 user embeddings
        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=user_vocab, mask_token=None),
            tf.keras.layers.Embedding(len(user_vocab) + 1, 64)
        ])

        # 📦 item embeddings
        self.item_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=item_vocab, mask_token=None),
            tf.keras.layers.Embedding(len(item_vocab) + 1, 64)
        ])

        # 🔥 ranking network
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        # 🎯 task
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features):
        user_embeddings = self.user_model(features["user_id"])
        item_embeddings = self.item_model(features["item_id"])

        return self.rating_model(
            tf.concat([user_embeddings, item_embeddings], axis=1)
        )

    def compute_loss(self, features, training=False):
        labels = features["interaction"]

        predictions = self({
            "user_id": features["user_id"],
            "item_id": features["item_id"]
        })

        return self.task(labels=labels, predictions=predictions)