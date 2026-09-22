"""Encode / decode the `cfg` query parameter of share links.

New links carry zlib-compressed compact JSON in URL-safe base64 (no padding),
which is roughly 3x shorter than the original format. Legacy links (plain
standard base64 of JSON, always starting with "eyJ") still decode.
"""
import base64
import json
import zlib


def encode_cfg(cfg):
    raw = json.dumps(cfg, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode().rstrip("=")


def decode_cfg(s):
    s = s.strip()
    padded = s + "=" * (-len(s) % 4)
    if s.startswith("eyJ"):  # legacy: base64(json)
        return json.loads(base64.b64decode(padded).decode())
    return json.loads(zlib.decompress(base64.urlsafe_b64decode(padded)).decode())
