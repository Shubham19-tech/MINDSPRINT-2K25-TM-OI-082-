
print("✔ Creating actions.py ...")

from fetch import final_col

def approve(candidate_name):
   print(f"Approving: {candidate_name}")
   final_col.update_one(
       {"candidate.name": candidate_name},
       {"$set": {"status": "Approved"}}
   )

def reject(candidate_name):
   print(f"Rejecting: {candidate_name}")
   final_col.update_one(
       {"candidate.name": candidate_name},
       {"$set": {"status": "Rejected"}}
   )

print("✔ actions.py created successfully!")
