
print("✔ Creating matching.py ...")

from scoring import total_score
from fetch import save_final_match

def get_resume_value(resume, keys, default=None):
   for k in keys:
       if k in resume and resume[k] not in [None, "", []]:
           return resume[k]
   return default

def parse_resume(res):
   parsed = {
       "name": get_resume_value(res, ["name", "candidateName", "fullName"], "Unknown Candidate"),
       "skills": get_resume_value(res, ["skills", "techSkills", "technologies", "skillset"], []),
       "experience": get_resume_value(res, ["experience", "exp", "years", "totalExperience"], 0),
       "job_role": get_resume_value(res, ["job_role", "role", "position", "jobTitle"], "Unknown Role"),
       "location": get_resume_value(res, ["location", "city", "currentLocation"], "Unknown"),
       "work_mode": get_resume_value(res, ["work_mode", "workMode", "preferredWorkMode"], "Not Given"),
       "raw_data": res
   }
   print(f"Parsed resume: {parsed['name']}")
   return parsed

def match_resumes_to_job(resumes, jobs):
   print("Starting matching engine...")

   parsed_resumes = [parse_resume(r) for r in resumes]
   results = []

   for job in jobs:
       job_title = job.get("title", "Unknown Job")
       print(f"\n🔍 Matching for job: {job_title}")

       shortlisted = []

       for res in parsed_resumes:
           score = total_score(res, job)
           print(f"Score for {res['name']}: {score}")

           if score >= 60:
               data = {
                   "job_title": job_title,
                   "score": score,
                   "candidate": res["raw_data"]
               }
               shortlisted.append(data)
               save_final_match(data)

       results.append({
           "job_title": job_title,
           "shortlisted": shortlisted
       })

   print("✔ Matching complete!")
   return results

print("✔ matching.py created successfully!")
