"""Snapshot and restore the three files under %APPDATA%\\ConvoPeq that the
measurement scripts overwrite. Use around every run:

    guard = profile_guard.Guard()
    try:
        guard.save()
        ...run...
    finally:
        guard.restore()
"""
import os
import shutil
import tempfile
import time

APPDATA = os.path.join(os.environ["APPDATA"], "ConvoPeq")
NAMES = ("device_settings.xml", "noise_shaper_learn.xml", "learned_state.xml")


class Guard:
    def __init__(self):
        self.dir = None
        self.saved = {}
        self.complete = False

    def _lock(self):
        self.lock = os.path.join(tempfile.gettempdir(), "m236profile.lock")
        for _ in range(600):  # up to 60 s
            try:
                handle = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(handle)
                return
            except FileExistsError:
                time.sleep(0.1)
        raise RuntimeError("another measurement run holds %s" % self.lock)

    def save(self):
        self._lock()
        self.dir = tempfile.mkdtemp(prefix="m236profile-")
        for name in NAMES:
            src = os.path.join(APPDATA, name)
            if os.path.exists(src):
                dst = os.path.join(self.dir, name)
                shutil.copyfile(src, dst)
                self.saved[name] = dst
        self.complete = True

    def restore(self):
        if self.dir is None:
            return
        for name in NAMES:
            dst = os.path.join(APPDATA, name)
            if name in self.saved:
                shutil.copyfile(self.saved[name], dst)
            elif self.complete and os.path.exists(dst):
                # only delete when the snapshot was complete: a partial save
                # must not be read as "the user did not have this file"
                os.remove(dst)
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir = None
        if getattr(self, "lock", None) and os.path.exists(self.lock):
            os.remove(self.lock)
            self.lock = None
