"""Platform-isolation probe for the compiled-model hash mismatch (Study 02E).

Compiles single upstream STL meshes inside a minimal, identical MJCF string and
hashes MuJoCo's compile-time arrays. Running it with the same MuJoCo version on
two platforms separates platform-dependent mesh processing from upstream source
bytes (line endings), task configuration or mjlab. It needs only ``mujoco`` and
``numpy``; it reads upstream meshes and writes one JSON file.

    python -m movement_lab.platform02e --meshes <upstream>/src/mjlab_microduck/robot/microduck --out x.json
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

MESHES = ('sole_left', 'sole_right', 'jaw', 'power_support', 'trunk_base')
FIELDS = ('mesh_vert', 'mesh_face', 'mesh_normal', 'mesh_graph', 'mesh_polyvert', 'mesh_polynormal', 'mesh_polymap')


def probe(mesh_dir):
    import mujoco
    import numpy as np
    out = {}
    for name in MESHES:
        found = sorted(Path(mesh_dir).rglob(f'{name}.stl'))
        if len(found) != 1:
            raise SystemExit(f'expected exactly one {name}.stl under {mesh_dir}, found {len(found)}')
        stl = found[0]
        xml = (f'<mujoco><asset><mesh name="m" file="{stl.resolve().as_posix()}"/></asset>'
               '<worldbody><body><freejoint/><geom type="mesh" mesh="m"/></body></worldbody></mujoco>')
        m = mujoco.MjModel.from_xml_string(xml)
        out[name] = dict(stl_sha256=hashlib.sha256(stl.read_bytes()).hexdigest(),
                         hull_vertices=int(m.mesh_graph[0]), hull_faces=int(m.mesh_graph[1]),
                         **{f: hashlib.sha256(np.ascontiguousarray(getattr(m, f)).tobytes()).hexdigest() for f in FIELDS})
    return dict(platform=platform.platform(), python=sys.version, mujoco=mujoco.__version__, meshes=out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--meshes', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f'refusing to overwrite {a.out}')
    result = probe(a.meshes)
    a.out.write_text(json.dumps(result, indent=1) + '\n', newline='\n')
    print(json.dumps({k: {f: v[f][:12] for f in ('mesh_vert', 'mesh_graph', 'mesh_polyvert')} for k, v in result['meshes'].items()}, indent=1))


if __name__ == '__main__':
    main()
