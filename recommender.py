
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from math import radians, sin, cos, sqrt, atan2

# DataFrames filled from main.py
users = None
preferences = None
interactions = None
services = None

model = None
model_ready = False


class RankingModel(tf.keras.Model):
    def __init__(self, user_model, item_model, rating_model):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.rating_model = rating_model

    def call(self, features):
        user_emb = self.user_model(features["user_id"])
        item_emb = self.item_model(features["item_id"])
        return self.rating_model([user_emb, item_emb])


class TFRSRanking(tfrs.models.Model):
    def __init__(self, ranking_model):
        super().__init__()
        self.ranking_model = ranking_model
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
        )

    def compute_loss(self, features, training=False):
        preds = self.ranking_model({
            "user_id": features["user_id"],
            "item_id": features["item_id"]
        })
        return self.task(labels=features["interaction"], predictions=preds)
from model_def import RecommendationModel


def build_model():
    global model, model_ready

    # vocab
    with open("user_vocab.json") as f:
        user_vocab = json.load(f)

    with open("item_vocab.json") as f:
        item_vocab = json.load(f)

    # 🔥 نفس مودل التدريب
    model = RecommendationModel(user_vocab, item_vocab)

    # build dummy
    dummy = {
        "user_id": tf.constant(["1"]),
        "item_id": tf.constant(["1"]),
        "interaction": tf.constant([1.0])
    }

    _ = model.compute_loss(dummy)

    # load weights
    model.load_weights("ckpt.weights.h5")

    model_ready = True
    print("🔥 Model loaded (same as training)")

def predict_score(user_id, item_id):
    global model, model_ready

    # 🔥 أهم سطر
    if not model_ready:
        build_model()

    score = model({
    "user_id": tf.constant([str(user_id)]),
    "item_id": tf.constant([str(item_id)])
})

    return float(score.numpy()[0][0])

def get_user_profile(user_id):
    user_id = int(user_id)

    user_row = users[users["user_id"].astype(int) == user_id]
    pref_row = preferences[preferences["user_id"].astype(int) == user_id]

    if len(user_row) == 0:
        return None

    has_preferences = len(pref_row) > 0
    has_history = len(interactions[interactions["user_id"].astype(int) == user_id]) > 0

    profile = {
        "user_id": user_id,
        "has_history": has_history,
        "has_preferences": has_preferences
    }

    if not has_preferences:
        profile.update({
            "distance_preference": "flexible",
            "food_preference": "unknown",
            "preferred_event_types": "unknown",
            "preferred_places": "unknown",
            "accommodation_type": "unknown",
            "budget_preference": "medium"
        })
    else:
        pref = pref_row.iloc[0]
        profile.update({
            "distance_preference": str(pref.get("distance_preference", "flexible")).lower().strip(),
            "food_preference": str(pref.get("food_preference", "unknown")).lower().strip(),
            "preferred_event_types": str(pref.get("preferred_event_types", "unknown")).lower().strip(),
            "preferred_places": str(pref.get("preferred_places", "unknown")).lower().strip(),
            "accommodation_type": str(pref.get("accommodation_type", "unknown")).lower().strip(),
            "budget_preference": str(pref.get("budget_preference", "medium")).lower().strip()
        })

    return profile

def merge_popularity(recs):
    service_popularity = get_service_popularity()
    recs["service_id"] = recs["service_id"].astype(str)
    recs = recs.merge(service_popularity, on="service_id", how="left")
    recs["popularity"] = recs["popularity"].fillna(0)
    return recs

def normalize_features(recs):
    recs = recs.copy()

    # 🔹 Normalize basic features
    recs["distance_norm"] = (
        recs["distance_km"] / recs["distance_km"].max()
        if recs["distance_km"].max() > 0 else 0
    )

    recs["rating_norm"] = (
        recs["rating"] / recs["rating"].max()
        if recs["rating"].max() > 0 else 0
    )

    recs["popularity_norm"] = (
        recs["popularity"] / recs["popularity"].max()
        if recs["popularity"].max() > 0 else 0
    )

    # 🔥 Handle predicted_score safely
    if "predicted_score" in recs.columns:

        # احسب المتوسط
        mean_score = recs["predicted_score"].mean()

        # لو كله None → fallback
        if pd.isna(mean_score):
            mean_score = 0.5

        # عبي القيم الفارغة
        recs["predicted_score"] = recs["predicted_score"].fillna(mean_score)

        # normalize predicted score
        recs["predicted_norm"] = (
            recs["predicted_score"] - recs["predicted_score"].min()
        ) / (
            recs["predicted_score"].max() - recs["predicted_score"].min() + 1e-8
        )

    return recs

def apply_distance_filter(profile, recs, max_distance_km):
    if profile["distance_preference"] == "nearby":
        filtered = recs[recs["distance_km"] <= max_distance_km]
        if len(filtered) > 10:
            return filtered

    elif profile["distance_preference"] == "moderate":
        filtered = recs[recs["distance_km"] <= max_distance_km * 1.5]
        if len(filtered) > 10:
            return filtered

    return recs
def score_services(user_id, df):
    df = df.copy()
    df["service_id"] = df["service_id"].astype(str)

    def safe_predict(item_id):
        if is_service_in_vocab(item_id):
            return predict_score(user_id, item_id)
        else:
            return None  # fallback

    df["predicted_score"] = df["service_id"].apply(safe_predict)

    return df


def add_distance(user_id, df):
    df = df.copy()

    user = users[users["user_id"].astype(int) == int(user_id)]
    if len(user) == 0:
        df["distance_km"] = 9999.0
        return df

    user = user.iloc[0]
    user_lat = float(user["latitude"])
    user_lon = float(user["longitude"])

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = (
            sin(dlat / 2) ** 2
            + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        )
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    distances = []
    for _, row in df.iterrows():
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            distances.append(haversine(user_lat, user_lon, lat, lon))
        except Exception:
            distances.append(9999.0)

    df["distance_km"] = distances
    return df


def get_service_popularity():
    pop = interactions.copy()
    pop["service_id"] = pop["service_id"].astype(str)
    pop = pop.groupby("service_id").size().reset_index(name="popularity")
    return pop


def add_preference_match_features(profile, recs):
    recs = recs.copy()

    def normalize_token(token):
        token = str(token).lower().strip()
        mapping = {
            "arabic": "saudi",
            "restaurants": "restaurant",
            "restaurant": "restaurant",
            "cafes": "cafe",
            "cafe": "cafe",
            "stadiums": "stadium",
            "stadium": "stadium",
            "football matches": "sports",
            "football match": "sports",
            "sports events": "sports",
            "cultural events": "cultural",
            "cultural event": "cultural",
        }
        return mapping.get(token, token)

    food_pref = str(profile.get("food_preference", "unknown")).lower().strip()
    food_tokens = [
        normalize_token(x)
        for x in food_pref.split(";")
        if x.strip() and x.strip() != "unknown"
    ]

    recs["food_match"] = recs.apply(
        lambda row: 1.0 if food_tokens and any(
            token in str(row.get("cuisine_type", "")).lower()
            for token in food_tokens
        )
        else 0.5 if food_tokens and any(
            token in str(row.get("tags", "")).lower()
            for token in food_tokens
        )
        else 0.0,
        axis=1
    )

    budget_pref = str(profile.get("budget_preference", "unknown")).lower().strip()
    recs["budget_match"] = recs["price_range"].fillna("").astype(str).str.lower().apply(
        lambda x: 1 if budget_pref not in ["", "unknown", "nan"] and budget_pref in x else 0
    )

    event_pref = str(profile.get("preferred_event_types", "unknown")).lower().strip()
    event_tokens = [
        normalize_token(x)
        for x in event_pref.split(";")
        if x.strip() and x.strip() != "unknown"
    ]

    if "event_type" in recs.columns:
        recs["event_match"] = recs["event_type"].fillna("").astype(str).str.lower().apply(
            lambda x: 1 if event_tokens and any(
                token in x or (token == "cultural" and x in ["festival", "exhibition"])
                for token in event_tokens
            ) else 0
        )
    else:
        recs["event_match"] = 0

    place_pref = str(profile.get("preferred_places", "unknown")).lower().strip()
    place_tokens = [
        normalize_token(x)
        for x in place_pref.split(";")
        if x.strip() and x.strip() != "unknown"
    ]

    recs["place_match"] = recs.apply(
        lambda row: 1 if place_tokens and any(
            token in str(row.get("tags", "")).lower()
            or token in str(row.get("service_category", "")).lower()
            for token in place_tokens
        ) else 0,
        axis=1
    )

    if "family_friendly" in recs.columns:
        recs["family_match"] = recs["family_friendly"].apply(
            lambda x: 1 if str(x).lower() == "yes" else 0
        )
    else:
        recs["family_match"] = 0

    return recs


def get_dynamic_weights(profile):
    w_pred = 0.70
    w_rating = 0.20
    w_pop = 0.10

    dist_pref = profile.get("distance_preference", "flexible")

    if dist_pref == "nearby":
        w_dist = 0.08
    elif dist_pref == "moderate":
        w_dist = 0.04
    else:
        w_dist = 0.01

    return w_pred, w_rating, w_pop, w_dist

def add_context_and_rerank(profile, recs, max_distance_km=10):
    recs = add_distance(profile["user_id"], recs)

    if len(recs) == 0:
        return recs

    # 🔥 بدل التكرار
    recs = apply_distance_filter(profile, recs, max_distance_km)

    recs = merge_popularity(recs)
    recs = add_preference_match_features(profile, recs)
    recs = normalize_features(recs)

    w_pred, w_rating, w_pop, w_dist = get_dynamic_weights(profile)

    recs["final_score"] = (
        w_pred * recs["predicted_norm"] +
        w_rating * recs["rating_norm"] +
        w_pop * recs["popularity_norm"] -
        w_dist * recs["distance_norm"] +
        0.10 * recs["food_match"] +
        0.05 * recs["budget_match"] +
        0.05 * recs["event_match"] +
        0.05 * recs["place_match"] +
        0.03 * recs["family_match"]
    )

    return recs.sort_values("final_score", ascending=False)
def recommend(user_id):
    print("TEST SCORE:", predict_score(user_id, "322"))
    
    print("INTERACTIONS COUNT:",
          len(interactions["service_id"].astype(str).unique()))

    print("SERVICES COUNT:", len(services))

    profile = get_user_profile(user_id)

    if profile is None:
        recs = popular_baseline(k=50, max_distance_km=None, user_id=None)
        recs = recs.sort_values("baseline_score", ascending=False)

        return {
            "for_you": clean_records(recs.head(15)),
            "all": clean_records(recs)
        }

    has_preferences = profile.get("has_preferences", False)
    has_history = profile.get("has_history", False)

    # 🔥 الحالة الأساسية (Hybrid)
    if has_preferences and has_history:

        # # ✅ 1. نفس الكولاب: فقط items اللي في interactions
        # valid_items = interactions["service_id"].astype(str).unique()

        # filtered_services = services[
        #     services["service_id"].astype(str).isin(valid_items)
        # ].copy()
        filtered_services = services.copy()

        # ✅ 2. حساب predicted_score
        recs = score_services(user_id, filtered_services)

        # 🔥🔥 أهم خطوة (كانت ناقصة عندك)
        # نفس سلوك الكولاب قبل أي معالجة
        recs = recs.sort_values("predicted_score", ascending=False)

        # ✅ 3. hybrid reranking
        recs = add_context_and_rerank(profile, recs, max_distance_km=10)

        # ✅ 4. الترتيب النهائي
        recs = recs.sort_values("final_score", ascending=False)

        # 🔥 For You (متنوع)
        hotels = recs[recs["service_category"] == "hotel"].head(5)
        restaurants = recs[recs["service_category"] == "restaurant"].head(5)
        events = recs[recs["service_category"] == "event"].head(5)

        balanced = pd.concat([hotels, restaurants, events])
        balanced = balanced.sort_values("final_score", ascending=False)

        return {
            "for_you": clean_records(balanced[[
                "service_id", "service_name", "service_category",
                "predicted_score", "predicted_norm", "final_score",
                "rating", "city", "price_range"
            ]]),

            "all": clean_records(recs[[
                "service_id", "service_name", "service_category",
                "predicted_score", "predicted_norm", "final_score",
                "rating", "city", "price_range"
            ]])
        }

    # 🟡 preferences only
    elif has_preferences and not has_history:
        recs = recommend_preferences_only(profile, max_distance_km=10)
        recs = recs.sort_values("final_score", ascending=False)

        return {
            "for_you": clean_records(recs.head(15)),
            "all": clean_records(recs)
        }

    # 🔵 interactions only
    elif not has_preferences and has_history:
        recs = recommend_interactions_only(profile)
        recs = recs.sort_values("final_score", ascending=False)

        return {
            "for_you": clean_records(recs.head(15)),
            "all": clean_records(recs)
        }

    # ⚪ cold start
    else:
        recs = popular_baseline(k=50, max_distance_km=10, user_id=user_id)
        recs = recs.sort_values("baseline_score", ascending=False)
        print("FINAL RECS LENGTH:", len(recs))

        return {
            "for_you": clean_records(recs.head(15)),
            "all": clean_records(recs)
        }
def popular_baseline(k=50, max_distance_km=None, user_id=None):
    recs = services.copy()
    recs["service_id"] = recs["service_id"].astype(str)

    service_popularity = get_service_popularity()
    recs = recs.merge(service_popularity, on="service_id", how="left")
    recs["popularity"] = recs["popularity"].fillna(0)

    recs["rating_norm"] = recs["rating"] / recs["rating"].max() if recs["rating"].max() > 0 else 0
    recs["popularity_norm"] = recs["popularity"] / recs["popularity"].max() if recs["popularity"].max() > 0 else 0

    recs["baseline_score"] = 0.6 * recs["rating_norm"] + 0.4 * recs["popularity_norm"]

    if user_id is not None:
      recs = add_distance(user_id, recs)

    if max_distance_km is not None:
        filtered = recs[recs["distance_km"] <= max_distance_km].copy()

        if len(filtered) == 0:
            return recs

        return filtered

    return recs.sort_values("baseline_score", ascending=False)

def recommend_preferences_only(profile, max_distance_km=10):
    recs = services.copy()
    recs = add_distance(profile["user_id"], recs)

    if len(recs) == 0:
        return recs

    recs = apply_distance_filter(profile, recs, max_distance_km)
    recs = merge_popularity(recs)
    recs = add_preference_match_features(profile, recs)
    recs = normalize_features(recs)

    dist_pref = profile.get("distance_preference", "flexible")
    w_dist = 0.08 if dist_pref == "nearby" else 0.04 if dist_pref == "moderate" else 0.01

    recs["final_score"] = (
        0.35 * recs["rating_norm"] +
        0.20 * recs["popularity_norm"] -
        w_dist * recs["distance_norm"] +
        0.20 * recs["food_match"] +
        0.10 * recs["budget_match"] +
        0.05 * recs["event_match"] +
        0.07 * recs["place_match"] +
        0.03 * recs["family_match"]
    )

    return recs.sort_values("final_score", ascending=False)
def is_user_in_vocab(user_id):
    with open("user_vocab.json") as f:
        user_vocab = json.load(f)
    return str(user_id) in user_vocab

def recommend_interactions_only(profile):
    user_id = profile["user_id"]

    if is_user_in_vocab(user_id):
        # 🟢 الحالة الطبيعية
        recs = score_services(user_id, services)
        recs = merge_popularity(recs)
        recs = normalize_features(recs)

        recs["final_score"] = (
            0.75 * recs["predicted_norm"] +
            0.15 * recs["rating_norm"] +
            0.10 * recs["popularity_norm"]
        )

    else:
        # 🔴 user جديد → استخدم interactions فقط
        user_interactions = interactions[
            interactions["user_id"].astype(int) == int(user_id)
        ]

        service_ids = user_interactions["service_id"].astype(str).unique()

        user_services = services[
            services["service_id"].astype(str).isin(service_ids)
        ]

        cuisines = user_services["cuisine_type"].dropna().unique()

        recs = services[
          (services["service_category"].isin(categories)) |
          (services["cuisine_type"].isin(cuisines))
].copy()
        recs = services[
            services["service_category"].isin(categories)
        ].copy()

        recs = merge_popularity(recs)
        recs = normalize_features(recs)

        recs["final_score"] = (
            0.6 * recs["rating_norm"] +
            0.4 * recs["popularity_norm"]
        )

    return recs.sort_values("final_score", ascending=False)

def clean_records(df):
    df = df.copy()
    df = df.replace({float("inf"): None, float("-inf"): None})
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")

def is_service_in_vocab(service_id):
    with open("item_vocab.json") as f:
        item_vocab = json.load(f)
    return str(service_id) in item_vocab