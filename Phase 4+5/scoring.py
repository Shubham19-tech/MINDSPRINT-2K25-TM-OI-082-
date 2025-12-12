
print("✔ Creating scoring.py ...")

def normalize_list(v):
   return v if isinstance(v, list) else ([] if v is None else [v])

def normalize_str(v):
   if isinstance(v, list): return str(v[0]).lower()
   if isinstance(v, str): return v.lower()
   return str(v).lower()

def normalize_num(v):
   try: return int(v)
   except: return 0

def skill_match(resume_skills, job_skills):
   resume_skills = normalize_list(resume_skills)
   job_skills = normalize_list(job_skills)
   if not job_skills:
       return 100
   matched = len(
       set([s.lower() for s in resume_skills]) &
       set([s.lower() for s in job_skills])
   )
   return (matched / len(job_skills)) * 100

def experience_match(resume_exp, min_exp, max_exp):
   re = normalize_num(resume_exp)
   mn = normalize_num(min_exp)
   mx = normalize_num(max_exp)
   if mn <= re <= mx:
       return 100
   if re < mn:
       return (re / mn) * 100 if mn > 0 else 50
   return 60

def job_title_match(resume_title, job_title):
   r = normalize_str(resume_title)
   j = normalize_str(job_title)
   if j in r:
       return 100
   if any(word in r for word in j.split()):
       return 60
   return 20

def total_score(resume, job):
   return round(
       skill_match(resume["skills"], job.get("skills")) * 0.6 +
       experience_match(resume["experience"], job.get("min_experience"), job.get("max_experience")) * 0.25 +
       job_title_match(resume["job_role"], job.get("title")) * 0.15,
       2
   )

print("✔ scoring.py created successfully!")
