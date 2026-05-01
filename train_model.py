# train_model.py

import tensorflow as tf
from data_loader import load_data, preprocess_data, build_vocab
from model_def import RecommendationModel
import json


# 🚀 1) تحميل البيانات
users, preferences, interactions, services = load_data()

# ⚙️ 2) تجهيز البيانات
interactions, services = preprocess_data(interactions, services)

# 🔤 3) vocab
user_vocab, item_vocab = build_vocab(interactions, services)


# 💾 حفظ vocab (مهم للـ backend)
with open("user_vocab.json", "w") as f:
    json.dump(user_vocab, f)

with open("item_vocab.json", "w") as f:
    json.dump(item_vocab, f)


# 🧠 4) تحويل البيانات إلى TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices({
    "user_id": interactions["user_id"].values,
    "item_id": interactions["item_id"].values,
    "interaction": interactions["interaction"].values,
})

dataset = dataset.shuffle(10000).batch(256).cache().prefetch(tf.data.AUTOTUNE)


# 🔥 5) إنشاء المودل
model = RecommendationModel(user_vocab, item_vocab)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003)
)


# 🏋️‍♀️ 6) تدريب المودل
print("🚀 Start training...\n")

history = model.fit(
    dataset,
    epochs=5   # تقدرين تزيدينها بعدين
)

print("\n✅ Training finished!")


# 💾 7) حفظ المودل
model.save_weights("ckpt.weights.h5")

print("💾 Model saved successfully!")