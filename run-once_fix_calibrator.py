
import sys
import pickle

sys.path.insert(0, ".")  # ensure calibrator.py is importable

# Step 1: load the broken pkl by stubbing the missing class
import builtins
_real_import = builtins.__import__

class _Stub:
    def __init__(self): pass
    def __setstate__(self, state): self.__dict__.update(state)

# Inject the stub so pickle can unpickle the old object
import types
fake_main = types.ModuleType("__main__")
fake_main.ConformalCalibrator = _Stub
sys.modules["__main__"] = fake_main

with open("results/CQR/calibrator_stage2_v2.pkl", "rb") as f:
    old_cal = pickle.load(f)

print("Old calibrator loaded:", old_cal.__dict__)

# Step 2: rebuild using the real importable class
from calibrator import ConformalCalibrator

new_cal = ConformalCalibrator()
new_cal.__dict__.update(old_cal.__dict__)  # copy all attributes across

with open("results/CQR/calibrator_stage2_v2.pkl", "wb") as f:
    pickle.dump(new_cal, f)

print("✅ Calibrator re-saved with importable class")
print("Quantiles:", sorted(new_cal.q_per_level.keys()))