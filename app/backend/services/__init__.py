"""Service layer: stateful orchestration over the pure ml/ layer.

Services own process-lifetime state (the loaded backbone, the product registry)
and are wired together once at startup in ``main.py``. The api/ layer talks only
to services, never to ml/ directly.
"""
