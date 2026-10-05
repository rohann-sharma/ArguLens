import urllib.request, json

text = (
    "According to a 2024 Harvard study of 20,000 teenagers, "
    "80% of teenagers who use Instagram experience depression. "
    "Therefore Instagram should be banned for people under 18."
)
data = json.dumps({"text": text}).encode("utf-8")
req = urllib.request.Request(
    "http://localhost:8001/api/analyze",
    data=data,
    headers={"Content-Type": "application/json"}
)
try:
    resp = urllib.request.urlopen(req).read().decode("utf-8")
    result = json.loads(resp)
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Error: {e}")
