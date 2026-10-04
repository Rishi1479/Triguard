"""Helper: python docs/phase_update.py <N> <STATUS>  — set 'Status:' line under Phase N in docs/PHASES.md."""
import re
import sys

n, status = sys.argv[1], sys.argv[2]
path = "docs/PHASES.md"
s = open(path).read()
s = re.sub(rf"(## Phase {n}: [^\n]*\nStatus: )[A-Z ]+", rf"\g<1>{status}", s)
open(path, "w").write(s)
