import urllib.request
import json
import time

# Give vite a moment to start
time.sleep(3)

print("Checking if Vite is serving on 5173...")
try:
    resp = urllib.request.urlopen("http://localhost:5173/")
    print(f"Vite root response: {resp.status}")
except Exception as e:
    print(f"Failed to reach Vite root: {e}")

print("\nChecking if Vite proxy works for /api/analyze...")
text = "The sky is blue today. Thus, climate change is a complete hoax."
data = json.dumps({"text": text}).encode("utf-8")
req = urllib.request.Request(
    "http://localhost:5173/api/analyze",
    data=data,
    headers={"Content-Type": "application/json"},
)

try:
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read().decode("utf-8"))
    print(f"Proxy response status: {resp.status}")
    print(f"Overall score returned: {result.get('overall_score')}")
    print(f"Assessment: {result.get('overall_assessment')}")
    print("Proxy is working!")
except Exception as e:
    print(f"Proxy test failed: {e}")
