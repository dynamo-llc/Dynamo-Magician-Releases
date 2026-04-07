"""
_wait_and_open.py  --  polls the server until it is ready, then opens the browser.
Called by run.bat in a background process.
"""
import sys
import time
import urllib.request
import urllib.error
import webbrowser

URL = "http://127.0.0.1:8000/health"
OPEN = "http://localhost:8000"
TIMEOUT = 120   # give up after 2 minutes
INTERVAL = 0.5  # poll every 500 ms

deadline = time.time() + TIMEOUT
while time.time() < deadline:
    try:
        with urllib.request.urlopen(URL, timeout=2) as r:
            if r.status == 200:
                webbrowser.open(OPEN)
                sys.exit(0)
    except Exception:
        pass
    time.sleep(INTERVAL)

# Server never came up — nothing to do
sys.exit(1)
