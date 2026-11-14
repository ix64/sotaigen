"""How the IPC tables are kept: one folder per input under a cache, baked once, under a lock.

A key covers the inputs and the code that bakes and reads the table (sotaigen.sources.bake_key), so
an entry is good for as long as it exists, and a failed bake leaves `<key>.failed.json`: baking again
would only spend the same minutes on the same failure. Delete the record to bake it again.
"""

import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path


def failure(cache, key):
    """Why the entry `key` has no table, or None."""
    record = cache / f"{key}.failed.json"
    return json.loads(record.read_text())["error"] if record.exists() else None


def baked(cache, key, bake, read):
    """(read(folder), None) for the entry `key`, baked first by bake(folder) if it isn't there, or
    (None, why) when its bake failed, now or before.

    A bake that raises RuntimeError or ValueError has failed for this input, and its folder is kept
    beside the entries as `<name>.failed` for its reports; anything else (an interruption, a
    programming error) says nothing about the input and is raised.
    """
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / key
    record = cache / f"{key}.failed.json"
    with (cache / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists() and not record.exists():
            temporary = Path(tempfile.mkdtemp(prefix=f"{key[:12]}-", dir=cache))
            try:
                bake(temporary)
                os.replace(temporary, destination)
            except BaseException as error:
                shutil.move(temporary, cache / (temporary.name + ".failed"))
                if not isinstance(error, RuntimeError | ValueError):
                    raise
                record.write_text(json.dumps({"error": str(error)}))
        if destination.exists():
            return read(destination), None
    return None, failure(cache, key)
