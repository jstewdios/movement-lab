"""Losslessly package the self-contained replay below the static-host file limit."""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path


def package(source: str):
    start = source.index('const DATA=') + len('const DATA=')
    end = source.index(';\nconst $=id=>', start)
    raw = source[start:end].encode('utf-8')
    data = json.loads(raw)
    zipped = gzip.compress(raw, mtime=0)
    assert gzip.decompress(zipped) == raw
    encoded = base64.b64encode(zipped).decode('ascii')
    expression = ("JSON.parse(await new Response(new Blob([Uint8Array.from(atob('" + encoded +
                  "'),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text())")
    prefix = source[:start]
    assert prefix.count('<script>') == 1
    prefix = prefix.replace('<script>', '<script type="module">')
    result = prefix + expression + source[end:]
    metadata = dict(lossless_roundtrip=True, trials=len(data['trials']), presets=len(data['presets']),
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    payload_sha256=hashlib.sha256(raw).hexdigest(),
                    hosted_sha256=hashlib.sha256(result.encode()).hexdigest(),
                    source_bytes=len(source.encode()), hosted_bytes=len(result.encode()),
                    encoding='gzip/base64; native browser DecompressionStream')
    return result, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('results/02d/report/replay.html'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--metadata', type=Path)
    a = p.parse_args()
    html, meta = package(a.input.read_text())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(html)
    if a.metadata:
        a.metadata.write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
