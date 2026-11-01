"""02A adapter to the unchanged validated 01C execution path."""
import sys
import numpy as np
from movement_lab import transition01c as runner
from movement_lab.explore02a import ACTIONS

def main():
    # Remove adapter arguments before delegating the existing parser.
    i = sys.argv.index('--coefficients')
    h, f = map(float, sys.argv[i+1:i+3]); del sys.argv[i:i+3]
    if (h, f) not in ACTIONS:
        raise ValueError('outside declared 02A action space')
    amp = dict(runner.SEQUENCES['flex_+1'])
    amp = {k: v*f for k, v in amp.items()}
    amp['head_yaw'] = .08*h
    runner.SEQUENCES['explore02a'] = amp
    sys.argv.extend(['--sequence', 'explore02a'])
    runner.main()

if __name__ == '__main__': main()
