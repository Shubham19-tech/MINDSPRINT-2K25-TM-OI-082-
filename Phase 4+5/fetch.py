
print("✔ Creating fetch.py ...")

from pymongo import MongoClient

client = MongoClient(
   "mongodb+srv://praveenpatel5957_db_user:cjh2uxF03uIjWvjg@user.dlurkbi.mongodb.net/mindsprint"
)
db = client["mindsprint"]

resumes_col = db["resumes"]
jobs_col = db["jobdescriptions"]
final_col = db["finalresumes"]

def fetch_resumes():
   print("Fetching resumes...")
   return list(resumes_col.find({}, {"_id": 0}))

def fetch_jobs():
   print("Fetching job descriptions...")
   return list(jobs_col.find({}, {"_id": 0}))

def save_final_match(data):
   print(f"Saving match for: {data.get('candidate', {}).get('name', 'Unknown')}")
   final_col.insert_one(data)

def fetch_final_resumes():
   print("Fetching final shortlisted resumes...")
   return list(final_col.find({}, {"_id": 0}))


print("✔ fetch.py created successfully!")
