"""Find processes that have headroom.exe open."""
import psutil

headroom_path = r'C:\VSC_Project\ConvoPeq\.venv\Scripts\headroom.exe'

found = False
for proc in psutil.process_iter(['pid', 'name', 'exe', 'open_files', 'cmdline']):
    try:
        if proc.info['exe'] and headroom_path.lower() in proc.info['exe'].lower():
            print(f'EXE MATCH: PID={proc.info["pid"]} NAME={proc.info["name"]}')
            found = True

        if proc.info['open_files']:
            for f in proc.info['open_files']:
                if f.path and headroom_path.lower() in f.path.lower():
                    print(f'FILE MATCH: PID={proc.info["pid"]} PATH={f.path}')
                    found = True

        if proc.info['cmdline']:
            cmd = ' '.join(proc.info['cmdline'])
            if 'headroom' in cmd.lower() and 'proxy' in cmd.lower():
                print(f'PROXY MATCH: PID={proc.info["pid"]} CMD={cmd[:200]}')
                found = True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

if not found:
    print('No headroom processes found via psutil')

# Also check what's on port 8787
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = s.connect_ex(('127.0.0.1', 8787))
s.close()
print(f'Port 8787 connect test: {"OPEN" if result == 0 else "CLOSED"}')
