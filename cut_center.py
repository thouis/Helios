import sys
import cv2
import re


iblock = slice(3249, 47465)
jblock = slice(5099, 34750)

core_size = 16000

centers = open(sys.argv[1]).readlines()
files = sys.argv[2:]

scalei = float(iblock.stop - iblock.start)
scalej = float(jblock.stop - jblock.start)

pat = re.compile('([0-9]+) CENTER \(([0-9]+), ([0-9]+)\) \(([0-9]+), ([0-9]+)\) (.+) (.+)\n')
for l in centers:
    m = pat.match(l)
    idx = int(m.group(1))
    ci = int(m.group(2))
    cj = int(m.group(3))
    si = int(m.group(4))
    sj = int(m.group(5))
    oi = float(m.group(6))
    oj = float(m.group(7))
    print idx, iblock.start + scalei * (ci + oi) / si, \
        jblock.start + scalej * (cj + oj) / sj
