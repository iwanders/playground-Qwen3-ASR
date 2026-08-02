 
import os
from pathlib import Path

"""
HF_HOME=/abs_path_to_hf_home_storage_dir/
"""

envfile = Path(__file__).parent / ".env"
if envfile.is_file():
    for l in envfile.open().readlines():
        k,v = l.strip().split("=")
        os.environ[k] = v
