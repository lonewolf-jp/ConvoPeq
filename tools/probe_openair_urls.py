"""
Probe OpenAIR download URLs to find the correct pattern.
"""
import urllib.request
import sys

BASE = 'https://www.openair.hosted.york.ac.uk'

# Test various URL patterns for known IR files
test_urls = [
    '/wp-content/uploads/impulse/Central_Hall.wav',
    '/wp-content/uploads/ir/Central_Hall.wav',
    '/sites/default/files/impulse/Central_Hall_York.wav',
    '/wp-content/uploads/2019/impulse/Central_Hall.wav',
    '/files/impulse/Central_Hall.wav',
    '/wp-content/uploads/impulse_responses/Central_Hall.wav',
]

for path in test_urls:
    url = BASE + path
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=8)
        print(f'✓ FOUND: {url} ({resp.status})')
        print(f'  Content-Type: {resp.headers.get("Content-Type", "?")}')
        print(f'  Size: {resp.headers.get("Content-Length", "?")}')
        sys.exit(0)
    except urllib.error.HTTPError as e:
        print(f'  {e.code}: {path}')
    except Exception as e:
        print(f'  ERROR: {path} -> {e}')

print('\nNone of the guessed URLs worked.')
print('The OpenAIR site likely uses a database-driven approach')
print('where each IR has a unique post ID and the WAV file is')
print('attached as a WordPress media entry.')
print('\nSuggested approach: manually download from the IR Data page.')
