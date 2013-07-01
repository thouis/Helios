import sys
import os

for f in sys.argv[1:]:
    if not os.path.exists(os.path.join("Alyssa_downscaled_32",
                                       os.path.basename(f))):
        print "bsub -q short_serial python preprocess_alyssa_stack.py %s Alyssa_downscaled_32" % (f,)
