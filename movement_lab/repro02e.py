"""Study 02E Phase 0: reproducibility gate before any new experiment.

Read-only toward 02B/02C/02D: historical sources, checkpoints, results and the
compiled-model hash guard are used unchanged; every output goes under ``--out``.
Three kinds of evidence are kept separate:

* ``artifacts``  -- published-hash verification and recomputation from saved data
                    (02B/02C/02D verifiers, the 02D evidence audit). No physics.
* ``resimulate`` -- fresh numerical reproduction: every one of the 1,044 02D
                    held-out evaluation rows is re-simulated with the hash-matching
                    model and compared with the published row for exact equality.
                    With ``--alternate``, the same trial is also run on a freshly
                    compiled model and trajectories/outcomes are compared.
* ``compile`` / ``compare`` -- the compiled-model hash investigation: a fresh
                    compile through unchanged ``model02b.compile_model`` into the
                    02E tree, then a field-by-field and per-mesh hull comparison.

    python -m movement_lab.repro02e identity  --out results/02e/phase0
    python -m movement_lab.repro02e artifacts --out results/02e/phase0
    python -m movement_lab.repro02e compile   --out results/02e/phase0
    python -m movement_lab.repro02e compare   --out results/02e/phase0
    python -m movement_lab.repro02e resimulate --out results/02e/phase0 --alternate results/02e/phase0/linux-compile
    python -m movement_lab.repro02e gate      --out results/02e/phase0
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

HIST_ROOT = Path('results/02d')
UPSTREAM = Path(os.environ.get('MICRODUCK_RL', Path.home() / 'microduck_rl'))
# Compile-time arrays derived from MuJoCo's convex-hull/polygon processing.
HULL_FIELDS = ('mesh_graph', 'mesh_polymap', 'mesh_polynormal', 'mesh_polyvert', 'mesh_polyvertadr', 'mesh_polyvertnum',
               'mesh_polyadr', 'mesh_polynum', 'mesh_polymapadr', 'mesh_polymapnum', 'mesh_graphadr', 'npolyvert', 'npolymap',
               'nmeshgraph', 'nmeshpoly', 'nmeshpolyvert', 'nmeshpolymap')
OUTCOME_KEYS = ('recovered_200hz', 'survived', 'physical_valid', 'failure_time_s', 'failure_cause', 'settled_200hz',
                'recovery_with_step', 'upright_200hz', 'quiet_supported_final_second')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canon(x):
    return json.loads(json.dumps(x))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=1, sort_keys=False) + '\n', newline='\n')


# ----------------------------------------------------------------------------- identity
def identity():
    """Model, upstream, BAM, controller, dependency and evaluation identities."""
    import bam
    motor = Path(bam.__file__).parent / 'params/xl330/m6.json'
    motor_canonical = hashlib.sha256(json.dumps(json.loads(motor.read_text()), sort_keys=True,
                                                separators=(',', ':')).encode()).hexdigest()
    saved = json.loads(Path('results/02b/model/model.json').read_text())
    git = lambda *a: subprocess.run(['git', '-C', str(UPSTREAM), *a], capture_output=True, text=True).stdout.strip()
    up_files = {}
    for rel, key in (('src/mjlab_microduck/robot/microduck/robot_groundcontact.xml', 'robot_xml_sha256'),
                     ('src/mjlab_microduck/robot/microduck_constants.py', 'constants_sha256')):
        raw = (UPSTREAM / rel).read_bytes()
        up_files[rel] = dict(raw_sha256=hashlib.sha256(raw).hexdigest(),
                             crlf_sha256=hashlib.sha256(raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest(),
                             historical_model_json=saved['upstream'][key], has_crlf=b'\r\n' in raw)
    ckpts = {str(p): digest(p) for p in [Path('results/02b/runs/ars-push-s1/thetas.npz'), Path('results/02c/runs/cem-s1/final.npz'),
                                         Path('results/02c/runs/cem-s2/final.npz'),
                                         *sorted(Path('results/02d/runs').glob('*/final.npz'))]}
    sources = {str(p): digest(p) for p in sorted(Path('movement_lab').glob('*02[bcd]*')) if p.is_file() and p.suffix == '.py'}
    evals = {str(p): digest(p) for p in sorted(Path('results/02d/eval').glob('*/identity.json'))}
    pkgs = {k: importlib.metadata.version(k) for k in ('mujoco', 'mjlab', 'mujoco-warp', 'warp-lang', 'torch',
                                                        'better-actuator-models', 'numpy', 'scipy')}
    return dict(
        platform=platform.platform(), python=sys.version, packages=pkgs,
        historical_model_packages=saved['packages'], packages_match_historical_model={k: pkgs.get(k) == v for k, v in saved['packages'].items()},
        model=dict(path='results/02b/model/model.mjb', sha256=digest('results/02b/model/model.mjb'),
                   published_sha256=saved['mjb_sha256'], matches=digest('results/02b/model/model.mjb') == saved['mjb_sha256'],
                   compiled_on=saved['python'], published_model_json_sha256=digest('results/02b/model/model.json')),
        upstream=dict(path=str(UPSTREAM), revision=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain')),
                      historical_revision=saved['upstream']['revision'], files=up_files),
        bam=dict(parameters_sha256=digest(motor), canonical_sha256=motor_canonical,
                 historical_02c=json.loads(Path('results/02c/preflight.json').read_text())['bam_parameter_canonical_sha256'],
                 historical_02d=json.loads(Path('results/02d/provenance.json').read_text())['bam_parameter_canonical_sha256']),
        checkpoints=ckpts, historical_sources=sources, evaluation_identities=evals)


# ----------------------------------------------------------------------------- artifacts
def artifacts():
    """Published-hash verification plus recomputation from saved data. No physics."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    out = {}
    for label, cmd in (('02b_published_hashes', ['movement_lab.publication02b', '--root', 'results/02b', '--verify']),
                       ('02c_published_hashes', ['movement_lab.publication02c', '--root', 'results/02c', '--verify']),
                       ('02d_published_hashes', ['movement_lab.validate02d', '--root', 'results/02d', '--verify'])):
        r = subprocess.run([sys.executable, '-m', *cmd], capture_output=True, text=True, env=env)
        out[label] = dict(command=' '.join(['python -m', *cmd]), returncode=r.returncode,
                          stdout=r.stdout.strip()[-2000:], stderr=r.stderr.strip()[-2000:], passed=r.returncode == 0)
    # The 02D audit function is pure; its CLI would rewrite results/02d/validation.json, so call it directly.
    from movement_lab.validate02d import audit
    started = time.perf_counter()
    try:
        result = audit(HIST_ROOT, 0)
        out['02d_evidence_audit'] = dict(result, passed=True, wall_s=time.perf_counter() - started,
                                         note='RNG/CEM reconstruction, seed separation, declared budgets and '
                                              'physics-rate re-audit of saved records; resimulation reported separately')
    except AssertionError as e:
        out['02d_evidence_audit'] = dict(passed=False, failed_check=str(e))
    out['passed'] = all(v['passed'] for v in out.values())
    return out


# ----------------------------------------------------------------------------- model comparison
def mesh_graph_segment(m, i):
    adr = int(m.mesh_graphadr[i])
    later = [int(m.mesh_graphadr[j]) for j in range(i + 1, m.nmesh) if m.mesh_graphadr[j] >= 0]
    return m.mesh_graph[adr:(later[0] if later else len(m.mesh_graph))]


def hull_sets(m, i):
    g = mesh_graph_segment(m, i)
    nv, nf = int(g[0]), int(g[1])
    verts = set(g[2 + nv:2 + 2 * nv].tolist())
    faces = {tuple(sorted(f)) for f in g[len(g) - 3 * nf:].reshape(nf, 3).tolist()}
    normals = np.asarray(m.mesh_polynormal).reshape(-1, 3)
    pv = np.asarray(m.mesh_polyvert).ravel()
    polys = set()
    for p in range(int(m.mesh_polyadr[i]), int(m.mesh_polyadr[i] + m.mesh_polynum[i])):
        a, n = int(m.mesh_polyvertadr[p]), int(m.mesh_polyvertnum[p])
        polys.add((tuple(np.round(normals[p], 9).tolist()), frozenset(pv[a:a + n].tolist())))
    return nv, nf, verts, faces, polys


def model_fields(m):
    out = {}
    for k in dir(m):
        if k.startswith('_'):
            continue
        try:
            v = getattr(m, k)
        except Exception:
            continue
        if isinstance(v, np.ndarray) or isinstance(v, (int, float)):
            out[k] = v
    for k in dir(m.opt):
        if not k.startswith('_') and not callable(getattr(m.opt, k)):
            out['opt.' + k] = np.asarray(getattr(m.opt, k))
    return out


def fields_digest(m, exclude=HULL_FIELDS):
    """SHA256 over every numeric model field except ``exclude`` (diagnostic identity only)."""
    h = hashlib.sha256()
    for k, v in sorted(model_fields(m).items()):
        if k in exclude:
            continue
        arr = np.ascontiguousarray(np.asarray(v))
        h.update(k.encode() + str(arr.dtype).encode() + str(arr.shape).encode() + arr.tobytes())
    return h.hexdigest()


def compare_models(path_a, path_b):
    import mujoco
    a, b = mujoco.MjModel.from_binary_path(str(path_a)), mujoco.MjModel.from_binary_path(str(path_b))
    fa, fb = model_fields(a), model_fields(b)
    identical, different = [], {}
    for k in sorted(set(fa) | set(fb)):
        x, y = fa.get(k), fb.get(k)
        if x is None or y is None:
            different[k] = 'missing'
            continue
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            different[k] = dict(shape=[list(x.shape), list(y.shape)])
        elif np.array_equal(x, y, equal_nan=x.dtype.kind == 'f'):
            identical.append(k)
        else:
            d = dict(elements_differing=int(np.count_nonzero(x != y)), elements=int(x.size))
            if x.dtype.kind == 'f':
                d['max_abs_difference'] = float(np.nanmax(np.abs(x - y)))
            different[k] = d
    names = bytes(a.names) == bytes(b.names)
    paths = bytes(a.paths) == bytes(b.paths) if hasattr(a, 'paths') else None
    collision_mesh = {int(a.geom_dataid[g]): a.geom(g).name for g in range(a.ngeom)
                      if (a.geom_contype[g] or a.geom_conaffinity[g]) and a.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH}
    meshes = []
    for i in range(a.nmesh):
        if a.mesh_graphadr[i] < 0:
            continue
        same_bytes = np.array_equal(mesh_graph_segment(a, i), mesh_graph_segment(b, i))
        ha, hb = hull_sets(a, i), hull_sets(b, i)
        if same_bytes and ha[4] == hb[4]:
            continue
        meshes.append(dict(mesh=a.mesh(i).name, collision_geom=collision_mesh.get(i), is_collision=i in collision_mesh,
                           hull_vertices=[ha[0], hb[0]], hull_faces=[ha[1], hb[1]], graph_bytes_equal=bool(same_bytes),
                           same_hull_vertex_set=ha[2] == hb[2], same_hull_face_set=ha[3] == hb[3],
                           same_polygon_set=ha[4] == hb[4]))
    non_hull = [k for k in different if k not in HULL_FIELDS]
    return dict(model_a=dict(path=str(path_a), sha256=digest(path_a), fields_digest_excluding_hull=fields_digest(a)),
                model_b=dict(path=str(path_b), sha256=digest(path_b), fields_digest_excluding_hull=fields_digest(b)),
                fields_compared=len(identical) + len(different), identical_fields=len(identical), different_fields=different,
                differences_only_in_hull_fields=not non_hull, names_equal=names, paths_equal=paths,
                nmesh=int(a.nmesh), meshes_with_hull_differences=meshes)


# ----------------------------------------------------------------------------- resimulation
SIM = ALT = THETA = BASIS = None


def init_worker(alternate):
    global SIM, ALT, THETA, BASIS
    from movement_lab.sim02b import Sim
    from movement_lab.sim02c import BASE
    from movement_lab.control02d import load_basis
    SIM = Sim()                                   # unchanged hash guard against results/02b/model/model.json
    ALT = Sim(alternate) if alternate else None   # the fresh compile, checked against its OWN model.json
    THETA = np.load(BASE)['final']
    BASIS = load_basis()
    if hasattr(os, 'nice'):
        os.nice(10)


def evaluate_with(sim, task):
    """Exactly eval02d.evaluate, with the simulator passed explicitly."""
    from movement_lab.sim02d import rollout
    from movement_lab.eval02d import metrics, effective_mode
    name, vec, design, mode, label, c = task
    s, r = rollout(sim, THETA, vec, c, design, record=True, terminate=False, mode=effective_mode(mode, c), basis=BASIS)
    metrics(s, r, c)
    s.update(controller=name, set=label, condition=asdict(c), design=asdict(design), mode=mode)
    return s, r


def first_time(mask, t):
    idx = np.flatnonzero(mask)
    return None if not len(idx) else float(t[idx[0]])


def resimulate_one(job):
    group, task, published = job
    started = time.perf_counter()
    s, r = evaluate_with(SIM, task)
    fresh = canon(s)
    out = dict(group=group, controller=task[0], set=task[4], seed=task[5].seed, exact_historical=fresh == published,
               mismatched_keys=sorted(k for k in set(fresh) | set(published) if fresh.get(k) != published.get(k)),
               recovered_200hz=fresh['recovered_200hz'], failure_time_s=fresh['failure_time_s'],
               first_nonfoot_contact_s=first_time(r['nonfoot'] > 0, r['time_s']),
               first_self_contact_s=first_time(r['self_contacts'] > 0, r['time_s']))
    if ALT is not None:
        sa, ra = evaluate_with(ALT, task)
        alt = canon(sa)
        diff = np.abs(r['qpos'] - ra['qpos']).max(axis=1) if r['qpos'].shape == ra['qpos'].shape else None
        out.update(alt_bitwise=bool(diff is not None and not diff.any()),
                   alt_first_divergence_s=None if diff is None else first_time(diff > 0, r['time_s']),
                   alt_max_abs_qpos=None if diff is None else float(diff.max()),
                   alt_outcomes_equal=all(alt.get(k) == fresh.get(k) for k in OUTCOME_KEYS),
                   alt_row_equal=alt == fresh, alt_recovered_200hz=alt['recovered_200hz'], alt_failure_time_s=alt['failure_time_s'])
    out['wall_s'] = time.perf_counter() - started
    return out


def resimulation_jobs():
    from movement_lab.eval02d import controllers, battery, shifts
    from movement_lab.eval02c import read_rows
    jobs = []
    for group, conds in (('grid', battery()), ('shifts', shifts())):
        for name, vec, design, mode in controllers(HIST_ROOT):
            rows = read_rows(HIST_ROOT / 'eval' / group / f'{name}.jsonl.gz')
            assert len(rows) == len(conds)
            for (label, c), row in zip(conds, rows):
                assert canon(asdict(c)) == row['condition'] and label == row['set'], 'declared condition mismatch'
                jobs.append((group, (name, vec, design, mode, label, c), row))
    return jobs


def resimulate(out, alternate, workers):
    from movement_lab.eval02c import write_gzip
    jobs = resimulation_jobs()
    dest = out / 'resimulation.jsonl.gz'
    if dest.exists():
        raise SystemExit(f'refusing to overwrite {dest}')
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(str(alternate) if alternate else None,)) as pool:
        for i, row in enumerate(pool.map(resimulate_one, jobs, chunksize=4)):
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f'{i + 1}/{len(jobs)} exact={sum(r["exact_historical"] for r in rows)}', flush=True)
    write_gzip(dest, rows)
    write_json(out / 'resimulation-run.json', dict(wall_s=time.perf_counter() - started, workers=workers,
                                                   alternate_model_dir=str(alternate) if alternate else None))
    return summarize(out)


def summarize(out):
    """Summary recomputed from the saved per-trial comparison rows (no simulation)."""
    from movement_lab.eval02c import read_rows
    summary = summarize_resimulation(read_rows(out / 'resimulation.jsonl.gz'))
    run = out / 'resimulation-run.json'
    if run.exists():
        summary['run'] = json.loads(run.read_text())
    write_json(out / 'resimulation-summary.json', summary)
    print(json.dumps(summary, indent=1)[:3000])
    return summary


def summarize_resimulation(rows):
    s = dict(rows=len(rows), exact_historical=sum(r['exact_historical'] for r in rows),
             by_group={g: dict(rows=sum(r['group'] == g for r in rows), exact=sum(r['exact_historical'] for r in rows if r['group'] == g))
                       for g in ('grid', 'shifts')})
    if rows and 'alt_bitwise' in rows[0]:
        div = [r for r in rows if not r['alt_bitwise']]
        def after(r):
            marks = [x for x in (r['failure_time_s'], r['first_nonfoot_contact_s'], r['first_self_contact_s']) if x is not None]
            return bool(marks) and r['alt_first_divergence_s'] is not None and r['alt_first_divergence_s'] >= min(marks)
        s['alternate'] = dict(bitwise_identical=len(rows) - len(div), diverged=len(div),
                              outcomes_equal=sum(r['alt_outcomes_equal'] for r in rows),
                              rows_equal=sum(r['alt_row_equal'] for r in rows),
                              recovered_historical=sum(r['recovered_200hz'] for r in rows),
                              recovered_alternate=sum(r['alt_recovered_200hz'] for r in rows),
                              divergence_after_failure_or_nonfoot_or_self_contact=sum(after(r) for r in div),
                              divergence_before_any_such_event=[dict(controller=r['controller'], set=r['set'], seed=r['seed'],
                                                                     t=r['alt_first_divergence_s']) for r in div if not after(r)],
                              earliest_divergence_s=min((r['alt_first_divergence_s'] for r in div), default=None),
                              outcome_differences=[dict(controller=r['controller'], set=r['set'], seed=r['seed'], group=r['group'],
                                                        recovered=[r['recovered_200hz'], r['alt_recovered_200hz']],
                                                        failure=[r['failure_time_s'], r['alt_failure_time_s']])
                                                   for r in rows if not r['alt_outcomes_equal']])
    return s


# ----------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=('identity', 'artifacts', 'compile', 'compare', 'resimulate', 'summarize', 'preflight', 'gate'))
    p.add_argument('--out', type=Path, default=Path('results/02e/phase0'))
    p.add_argument('--alternate', type=Path, default=None, help='directory with a freshly compiled model.mjb + its model.json')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.step == 'identity':
        write_json(a.out / 'identity.json', identity())
    elif a.step == 'artifacts':
        result = artifacts()
        write_json(a.out / 'artifacts.json', result)
        print(json.dumps({k: v['passed'] for k, v in result.items() if isinstance(v, dict)}, indent=1))
    elif a.step == 'compile':
        from movement_lab.model02b import compile_model
        dest = a.out / 'linux-compile'
        dest.mkdir(exist_ok=False)
        info = compile_model(dest, UPSTREAM)  # unchanged historical compiler, separate output tree
        print('compiled', info['mjb_sha256'])
    elif a.step == 'compare':
        result = compare_models(Path('results/02b/model/model.mjb'), a.out / 'linux-compile/model.mjb')
        platforms = {}
        for f in sorted(a.out.glob('platform-*.json')):
            platforms[f.stem] = json.loads(f.read_text())
        if len(platforms) >= 2:
            (n1, p1), (n2, p2) = list(platforms.items())[:2]
            result['single_mesh_platform_probe'] = {
                mesh: {field: p1['meshes'][mesh][field] == p2['meshes'][mesh][field]
                       for field in p1['meshes'][mesh] if field not in ('hull_vertices', 'hull_faces')}
                for mesh in p1['meshes']}
            result['single_mesh_platform_probe_files'] = [n1, n2]
        write_json(a.out / 'model-comparison.json', result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('meshes_with_hull_differences',)}, indent=1)[:4000])
    elif a.step == 'resimulate':
        resimulate(a.out, a.alternate, a.workers)
    elif a.step == 'summarize':
        summarize(a.out)
    elif a.step == 'preflight':
        # Re-run 02D's unchanged preflight (120 historical 02C outcomes, zero-residual exactness,
        # determinism, passive recording, BAM parameters) into the 02E tree.
        root = a.out / 'preflight02d-rerun'
        root.mkdir(exist_ok=False)  # preflight02d itself refuses to overwrite but does not create its root
        subprocess.run([sys.executable, '-m', 'movement_lab.preflight02d', '--root', str(root)], check=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    else:
        gate(a.out)


def gate(out):
    ident = json.loads((out / 'identity.json').read_text())
    art = json.loads((out / 'artifacts.json').read_text())
    pre = json.loads((out / 'preflight02d-rerun/preflight.json').read_text())
    res = json.loads((out / 'resimulation-summary.json').read_text())
    cmp_ = json.loads((out / 'model-comparison.json').read_text())
    checks = dict(
        model_hash_matches_published=ident['model']['matches'],
        upstream_revision_matches=ident['upstream']['revision'] == ident['upstream']['historical_revision'] and not ident['upstream']['dirty'],
        bam_parameters_match_02c_02d=ident['bam']['canonical_sha256'] == ident['bam']['historical_02c'] == ident['bam']['historical_02d'],
        packages_match_historical_model=all(ident['packages_match_historical_model'].values()),
        published_artifacts_verified=art['passed'],
        historical_02c_outcomes_reproduced=pre['passed'] and pre['outcome_matches'] == 120,
        all_02d_heldout_rows_resimulated_exactly=res['exact_historical'] == res['rows'] == 1044)
    findings = dict(fresh_compile_sha256=cmp_['model_b']['sha256'],
                    fresh_compile_differs_only_in_hull_fields=cmp_['differences_only_in_hull_fields'],
                    alternate_model=res.get('alternate'))
    result = dict(passed=all(checks.values()), checks=checks, model_investigation=findings,
                  decision='02E simulations use the hash-matching historical model through the unchanged guard. '
                           'The fresh Linux compile is NOT substituted for it.')
    write_json(out / 'gate.json', result)
    print(json.dumps(result, indent=1))
    if not result['passed']:
        raise SystemExit('Phase 0 gate failed')


if __name__ == '__main__':
    main()
