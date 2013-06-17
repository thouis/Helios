import sys

def w(f1, f2, idx1, idx2):
    tmp = "%02d_%02d" % (idx1, idx2)
    print "bsub -q short_serial -o logs/out.%s -e logs/error.%s python helios.py %s %s warps/%s.hdf5" % (tmp, tmp, f1, f2, tmp)

for idx1, (f1, f2) in enumerate(zip(sys.argv[1:], sys.argv[2:])):
    w(f1, f2, idx1 + 1, idx1 + 2)
    w(f2, f1, idx1 + 2, idx1 + 1)

for idx1, (f1, f2) in enumerate(zip(sys.argv[1:], sys.argv[3:])):
    w(f1, f2, idx1 + 1, idx2 + 3)
    w(f2, f1, idx1 + 3, idx2 + 1)

