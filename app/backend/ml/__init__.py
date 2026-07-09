"""Pure ML layer: backbone, image transforms, evidential math.

Nothing here knows about HTTP, FastAPI, or persistence. Everything is a plain
function or a small class over tensors/arrays, so it can be unit-tested and
reused (e.g. from a notebook) in isolation.
"""
