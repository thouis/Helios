import sys
import os

def w(f1, f2, idx1, idx2):
    tmp = "%02d_%02d" % (idx1, idx2)
    if not os.path.exists("warps/%s.hdf5" % (tmp)):
        print 'bsub -q short_serial -R "rusage[mem=10000]" -g /alyssa_stack -o logs/out.%s -e logs/error.%s python helios.py %s %s warps/%s.hdf5' % (tmp, tmp, f1, f2, tmp)

for idx1, (f1, f2) in enumerate(zip(sys.argv[1:], sys.argv[2:])):
    w(f1, f2, idx1 + 1, idx1 + 2)
    w(f2, f1, idx1 + 2, idx1 + 1)

for idx1, (f1, f2) in enumerate(zip(sys.argv[1:], sys.argv[3:])):
    w(f1, f2, idx1 + 1, idx1 + 3)
    w(f2, f1, idx1 + 3, idx1 + 1)

for idx1, (f1, f2) in enumerate(zip(sys.argv[1:], sys.argv[4:])):
    w(f1, f2, idx1 + 1, idx1 + 4)
    w(f2, f1, idx1 + 4, idx1 + 1)

for idx1, (f1, f2) in enumerate(zip(sys.argv[1:], sys.argv[5:])):
    w(f1, f2, idx1 + 1, idx1 + 5)
    w(f2, f1, idx1 + 5, idx1 + 1)
