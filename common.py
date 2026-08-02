 
import os
from pathlib import Path

envfile = Path(__file__).parent / ".env"
if envfile.is_file():
    for l in envfile.open().readlines():
        k,v = l.strip().split("=")
        os.environ[k] = v
