"""HTTP layer: routing and request/response handling only.

Routes translate HTTP <-> service calls and map exceptions to status codes.
They hold no business logic — that lives in services/.
"""
