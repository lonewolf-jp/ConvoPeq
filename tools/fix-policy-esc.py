# 一時修正スクリプト: requiredCrossfadeAuthorityApplications の pathRegex に AudioEngine.h を追加
import json
p = ".github/isr-ai-governance-policy.json"
d = json.load(open(p, encoding="utf-8"))
apps = d["residencyTelemetryChecks"]["requiredCrossfadeAuthorityApplications"]
old = r"^src/audioengine/(AudioEngine\.Commit\.cpp|AudioEngine\.Init\.cpp|AudioEngine\.Processing\.PrepareToPlay\.cpp|AudioEngine\.Processing\.ReleaseResources\.cpp|AudioEngine\.Timer\.cpp)$"
new = r"^src/audioengine/(AudioEngine\.h|AudioEngine\.Commit\.cpp|AudioEngine\.Init\.cpp|AudioEngine\.Processing\.PrepareToPlay\.cpp|AudioEngine\.Processing\.ReleaseResources\.cpp|AudioEngine\.Timer\.cpp)$"
fixed = 0
for app in apps:
    if app.get("pathRegex") == old:
        app["pathRegex"] = new
        fixed += 1
assert fixed == 1, f"expected 1, found {fixed}"
json.dump(d, open(p, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
print(f"FIXED {fixed} crossfade pathRegex")
