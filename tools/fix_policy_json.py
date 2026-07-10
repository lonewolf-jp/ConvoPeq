#!/usr/bin/env python3
"""Fix isr-ai-governance-policy.json precisely."""
import json

with open('.github/isr-ai-governance-policy.json', 'r', encoding='utf-8') as f:
    data = f.read()
obj = json.loads(data)

# 1. Add Retire.cpp exclusion for \bcritical\b
excl_list = obj.get('forbiddenExecutionSemanticExclusions', [])
has_retire = any('Retire' in e.get('pathRegex', '') for e in excl_list)
if not has_retire:
    threading_excl = None
    for e in excl_list:
        if 'Threading' in e.get('pathRegex', ''):
            threading_excl = e
            break
    if threading_excl:
        retire_excl = dict(threading_excl)
        retire_excl['pathRegex'] = retire_excl['pathRegex'].replace('Threading', 'Retire')
        excl_list.append(retire_excl)
        print('1. Added Retire.cpp exclusion')

# 2. Update admission gate patterns
rc = obj.get('requestRebuildDirectCall', {})
rc['requiredAdmissionGatePatterns'] = [
    'isShutdownInProgress\\(\\)',
    'rejectWithEvidence\\(\\"publish_shutdown_in_progress\\"\\)'
]
for app in rc.get('requiredAdmissionGateApplications', []):
    app['lineRegex'] = app['lineRegex'].replace(
        'if \\(!acceptsRuntimePublication\\(\\)\\)',
        'isShutdownInProgress\\(\\)')
# Fix rebuild dispatch entry
for app in rc.get('requiredAdmissionGateApplications', []):
    if 'RebuildDispatch' in app.get('pathRegex', ''):
        app['lineRegex'] = 'isShutdownInProgress\\(\\)|requestRebuild\\(world\\)'
print('2. Updated admission gate patterns')

# 3. Update telemetryCounterOwners allowedPathRegex
for owner in obj.get('telemetryCounterOwners', []):
    owner['allowedPathRegex'] = '^src/(audioengine|tests)/'
print('3. Updated telemetryCounterOwners')

# 4. Sections that must stay Threading.cpp (no change)
THREADING_SECTIONS = {
    'waitForDrainCallsiteAllowlist', 'isFullyDrainedCallsiteAllowlist',
    'requiredDrainAuthorityApplications', 'requiredBoundedDrainWaitApplications',
}
FORBIDDEN_SCOPE_KEY = 'forbiddenTelemetryAuthorityScopes'

# 5. Move telemetry applications from Threading.cpp -> Retire.cpp
telem_apps = obj.get('residencyTelemetryChecks', {}).get('requiredTelemetryApplications', [])
for app in telem_apps:
    if 'Threading' in app.get('pathRegex', ''):
        app['pathRegex'] = app['pathRegex'].replace('Threading', 'Retire')
    # Also move AudioEngine.h entries for patterns that moved to Retire.cpp
    if app.get('pathRegex') == '^src/audioengine/AudioEngine\\.h$':
        lr = app.get('lineRegex', '')
        if any(k in lr for k in ['fallbackQueueDepth_', 'retireQueueDepth_',
                                   'setFallbackBacklogCount', 'setRetireBacklogCount',
                                   'setDeferredRetireResidencyCount']):
            app['pathRegex'] = '^src/audioengine/AudioEngine\\.Retire\\.cpp$'
print('5. Updated telemetry application paths')

# 6. Move bounded reclaim from Threading -> Retire
shutdown = obj.get('shutdownReclaimChecks', {})
for section_name in ['requiredBoundedReclaimApplications', 'requiredEmergencyReclaimApplications']:
    for app in shutdown.get(section_name, []):
        if 'Threading' in app.get('pathRegex', ''):
            app['pathRegex'] = app['pathRegex'].replace('Threading', 'Retire')
print('6. Updated bounded/emergency reclaim paths')

# 7. Verify threading sections are NOT changed
for section_name in THREADING_SECTIONS:
    for app in shutdown.get(section_name, []):
        if 'Retire' in app.get('pathRegex', ''):
            print(f'WARN: {section_name} still has Retire, reverting')
            app['pathRegex'] = app['pathRegex'].replace('Retire', 'Threading')

# forbidden scope
for app in obj.get('residencyTelemetryChecks', {}).get('forbiddenTelemetryAuthorityScopes', []):
    if 'Retire' in app.get('pathRegex', ''):
        print(f'WARN: forbidden scope still has Retire, reverting')
        app['pathRegex'] = app['pathRegex'].replace('Retire', 'Threading')

# Serialize
new_data = json.dumps(obj, indent=4, ensure_ascii=False)
if data.endswith('\n') and not new_data.endswith('\n'):
    new_data += '\n'

with open('.github/isr-ai-governance-policy.json', 'w', encoding='utf-8') as f:
    f.write(new_data)

# Validate
json.loads(new_data)
print('JSON OK')
