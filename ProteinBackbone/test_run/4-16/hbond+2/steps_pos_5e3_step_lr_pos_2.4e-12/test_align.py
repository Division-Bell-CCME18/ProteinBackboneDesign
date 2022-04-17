import os
from pymol import cmd


cmd.align('1NWZ_A', '1OJH_A', cycles=0, object='aln')

