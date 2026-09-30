#python3 helper_scripts/run_helper_scripts.py
import subprocess
import os
os.makedirs("helper_scripts/output", exist_ok=True)
subprocess.run(["python3", "helper_scripts/list_movies.py"])
subprocess.run(["python3", "helper_scripts/list_movies_with_genres.py"])
subprocess.run(["python3", "helper_scripts/list_all_existing_genres.py"])
