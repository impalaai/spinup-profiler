"""Minimal HTTP helpers — stdlib only, never raises for a bad response."""

import json as _json
import urllib.error
import urllib.request


def request(url, method="GET", body=None, timeout=10, headers=None):
    """Returns (status|None, body_bytes, error_str|None). status None => transport error."""
    hdrs = dict(headers or {})
    data = None
    if body is not None:
        data = _json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    elif method in ("POST", "PUT"):
        data = b""
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read(), None
    except urllib.error.HTTPError as e:
        try:
            payload = e.read()
        except Exception:
            payload = b""
        return e.code, payload, None
    except Exception as e:  # URLError, socket.timeout, ConnectionReset, ...
        return None, b"", "%s: %s" % (type(e).__name__, e)


def json_body(raw):
    try:
        return _json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return None


def ok(status):
    return status is not None and 200 <= status < 300


def snippet(raw, n=200):
    return raw.decode("utf-8", "replace").strip()[:n]
