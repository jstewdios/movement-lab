"""Optional historical evidence, with strict checks for every available artifact."""
import json
from pathlib import Path
import numpy as np
from movement_lab.evaluate import sha

HISTORICAL = Path('results/01c/pilot-gpu/baseline')


class HistoricalMismatch(AssertionError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__('historical evidence mismatch: '+json.dumps(audit))


def compare_historical(baseline, historical=HISTORICAL):
    baseline, historical = Path(baseline), Path(historical)
    files = {'historical_arrays_exact': 'zero/trial.npz',
             'model_equal': 'replay_model.mjb', 'config_equal': 'resolved_config.txt'}
    missing = [name for name in files.values() if not (historical/name).is_file()]
    audit = dict(status='unavailable' if len(missing) == len(files) else 'incomplete' if missing else 'passed',
                 historical_root=str(historical), missing=missing, comparisons={})
    for key, name in files.items():
        if name in missing: continue
        # Absence of current evidence is always an error, never an optional check.
        if name.endswith('.npz'):
            with np.load(baseline/name) as new, np.load(historical/name) as old:
                equal = set(new) == set(old) and all(np.array_equal(new[k], old[k]) for k in old)
        else:
            equal = sha(baseline/name) == sha(historical/name)
        audit['comparisons'][key] = bool(equal)
    if not all(audit['comparisons'].values()):
        audit['status'] = 'failed'
        raise HistoricalMismatch(audit)
    return audit


def record_historical(baseline, destination, historical=HISTORICAL):
    try:
        audit = compare_historical(baseline, historical)
    except HistoricalMismatch as exc:
        Path(destination).write_text(json.dumps(exc.audit, indent=2)+'\n')
        raise
    Path(destination).write_text(json.dumps(audit, indent=2)+'\n')
    return audit


def describe_historical(audit):
    if audit['status'] == 'passed':
        return 'Historical 01C comparison passed: trajectory arrays, model and resolved configuration match exactly.'
    return (f"Historical 01C comparison {audit['status']}: historical validation was not established. "
            f"Missing artifacts: {', '.join(audit['missing'])}. "
            f"Available-artifact comparisons: {json.dumps(audit['comparisons'])}. "
            'New 02A records were validated independently against their own baseline.')
