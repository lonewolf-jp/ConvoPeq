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

APPDATA = os.path.join(os.environ["APPDATA"], "ConvoPeq")
NAMES = ("device_settings.xml", "noise_shaper_learn.xml", "learned_state.xml")


class Guard:
    def __init__(self):
        self.dir = None
        self.saved = {}

    def save(self):
        self.dir = tempfile.mkdtemp(prefix="m236profile-")
        for name in NAMES:
            src = os.path.join(APPDATA, name)
            if os.path.exists(src):
                dst = os.path.join(self.dir, name)
                shutil.copyfile(src, dst)
                self.saved[name] = dst

    def restore(self):
        if self.dir is None:
            return
        for name in NAMES:
            dst = os.path.join(APPDATA, name)
            if name in self.saved:
                shutil.copyfile(self.saved[name], dst)
            elif os.path.exists(dst):
                os.remove(dst)
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir = None
