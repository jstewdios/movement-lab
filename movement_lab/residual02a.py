"""02A adapter to the unchanged validated 01C execution path."""
import argparse
import sys
from movement_lab import transition01c as runner
from movement_lab.explore02a import ACTIONS

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--coefficients', type=float, nargs=2, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    h, f = args.coefficients
    if (h, f) not in ACTIONS:
        raise ValueError('outside declared 02A action space')
    amp = dict(runner.SEQUENCES['flex_+1'])
    amp = {k: v*f for k, v in amp.items()}
    amp['head_yaw'] = .08*h
    runner.SEQUENCES['explore02a'] = amp
    sys.argv = [sys.argv[0], '--out', args.out, '--seed', str(args.seed), '--sequence', 'explore02a']
    runner.main()

if __name__ == '__main__': main()
