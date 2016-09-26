import sys
import types
RTOL = 1e-5
ATOL = 1e-5
TEST_SEED = 310516

PY2 = sys.version_info[0] == 2

if PY2:
    CLASSTYPES = (classobj, type)
else:
    CLASSTYPES = (type,)