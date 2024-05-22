import importlib


class LazyVar:
    def __init__(self, fn):
        self.fn = fn
        self.var = None

    def __getattr__(self, attr):
        if self.var is None:
            print("call fn()")
            self.var = self.fn()
        return getattr(self.var, attr)


class Arm:
    def __init__(self, name, shoulder_offset=None):
        self.name = name
        assert self.name == "l_arm" or self.name == "r_arm"


# example
# old = LazyVar(lambda :importlib.import_module("reachy2_modelling.old"))
import reachy2_modelling.old
import reachy2_modelling.pin
import reachy2_modelling.symik

# import reachy2_modelling.ik
