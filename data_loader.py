# data_loader.py

from supabase import create_client
import pandas as pd

# 🔐 بيانات Supabase
SUPABASE_URL = "https://iuklbmmnetrozdppaohm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml1a2xibW1uZXRyb3pkcHBhb2htIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njk3MzQ1NzgsImV4cCI6MjA4NTMxMDU3OH0.x1eF_b8AMdFpzj4DnS0eEbfKJ-hccIgpIvIN2fNtrS0"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# 🔁 تحميل البيانات على دفعات (حل مشكلة التعليق)
def fetch_all(table_name, batch_size=1000):
    all_data = []
    start = 0

    while True:
        print(f"📥 Fetching {table_name}: rows {start} → {start + batch_size}")

        response = supabase.table(table_name) \
            .select("*") \
            .range(start, start + batch_size - 1) \
            .execute()

        data = response.data

        if not data:
            break

        all_data.extend(data)
        start += batch_size

    print(f"✅ Finished loading {table_name}: {len(all_data)} rows\n")
    return pd.DataFrame(all_data)


# 📥 تحميل كل البيانات
def load_data():
    print("🚀 Start loading data...\n")

    users = fetch_all("users", 500)
    preferences = fetch_all("preferences", 500)
    interactions = fetch_all("interactions", 1000)  # أكبر جدول
    services = fetch_all("services", 500)

    print("🎉 All data loaded successfully!\n")

    return users, preferences, interactions, services


# 🧠 تجهيز البيانات للمودل
def preprocess_data(interactions, services):

    print("⚙️ Preprocessing data...")

    # 🔥 تحويل interaction إلى رقم
    event_weight = {
        "viewed": 1.0,
        "clicked": 3.0,
        "bookmarked": 6.0,
        "reviewed": 10.0
    }

    interactions["interaction"] = interactions["interaction_type"].map(event_weight)

    # 🔥 توحيد الأنواع (مهم جدًا)
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["service_id"].astype(str)
    services["service_id"] = services["service_id"].astype(str)

    print("✅ Preprocessing done\n")

    return interactions, services


# 🔤 بناء vocab
def build_vocab(interactions, services):

    print("🔤 Building vocab...")

    user_vocab = sorted(interactions["user_id"].unique().tolist())

    item_vocab = sorted(
        set(interactions["item_id"].unique()).union(
            set(services["service_id"].unique())
        )
    )

    print(f"👤 Users: {len(user_vocab)}")
    print(f"📦 Items: {len(item_vocab)}\n")

    return user_vocab, item_vocab