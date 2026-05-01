from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
import pandas as pd
import recommender
from recommender import recommend as get_recommendations
import subprocess
from fastapi import BackgroundTasks
from recommender import recommend


SUPABASE_URL = "https://iuklbmmnetrozdppaohm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml1a2xibW1uZXRyb3pkcHBhb2htIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njk3MzQ1NzgsImV4cCI6MjA4NTMxMDU3OH0.x1eF_b8AMdFpzj4DnS0eEbfKJ-hccIgpIvIN2fNtrS0"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # 🔥 هذا المهم

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "YAHALA recommendation API is running"}

def load_data():
    users = pd.DataFrame(supabase.table("users").select("*").execute().data)
    preferences = pd.DataFrame(supabase.table("preferences").select("*").execute().data)
    interactions = pd.DataFrame(supabase.table("interactions").select("*").execute().data)
    services = pd.DataFrame(supabase.table("services").select("*").execute().data)
    return users, preferences, interactions, services

@app.get("/recommend/{user_id}")
def recommend_api(user_id: int):
    print("🚀 API HIT for user:", user_id)  # 🔥 مهم جدًا

    try:
        results = recommend(user_id)

        print("✅ RECOMMEND DONE")  # 🔥 تأكيد

        return {
            "user_id": user_id,
            "for_you": results["for_you"],
            "all": results["all"]
        }

    except Exception as e:
        print("❌ ERROR:", e)  # 🔥 يطبع الخطأ
        return {
            "user_id": user_id,
            "error": str(e)
        }
        
        
@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):

    def run_training():
        global model_ready

        print("🚀 Retraining started...")
        subprocess.run(["python", "train_model.py"])
        
        # 🔥 مهم جدًا
        import recommender
        recommender.model_ready = False

        print("✅ Retraining finished!")

    background_tasks.add_task(run_training)

    return {"message": "Retraining started"}
@app.on_event("startup")
def startup_event():
    print("🚀 Loading model at startup...")

    u, p, i, s = load_data()

    import recommender
    recommender.users = u
    recommender.preferences = p
    recommender.interactions = i
    recommender.services = s

    # 🔥 شغلي توصية وهمية (عشان يسخن المودل)
    try:
        recommend(1)
    except:
        pass

    print("✅ Model ready!")