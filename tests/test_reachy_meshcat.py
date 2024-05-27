from pinocchio.visualize import MeshcatVisualizer

try:
    import PyKDL as kdl  # NOQA
except ImportError as e:
    print("Error:", e)
    print(
        "need PyKDL you can install it from the conda env (see env.yaml) or from apt with:"
    )
    print("sudo apt install python3-pykdl")
    exit(1)

try:
    import numpy as np

    import reachy2_modelling as r2

except ImportError as e:
    print("Error:", e)
    print("is reachy2_modelling installed? run in root directory:")
    print("pip install -e .")
    exit(1)

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


class BaseMeshcat:
    def meshcat_collision(self):
        qq = self.q
        wrapper = self.wrapper
        viz = MeshcatVisualizer(
            wrapper.model, wrapper.model_collision(), wrapper.model_collision()
        )
        viz.initViewer(open=True)
        viz.loadViewerModel()
        viz.display(qq)

    def meshcat_visual(self):
        qq = self.q
        wrapper = self.wrapper
        viz = MeshcatVisualizer(
            wrapper.model, wrapper.model_collision(), wrapper.model_collision()
        )
        viz.initViewer(open=True)
        viz.loadViewerModel()
        viz.display(qq)


class TestReachy_Meshcat_LeftArm(BaseMeshcat):
    q = np.random.rand(7)
    wrapper = r2.pin.l_arm

    def test_meshcat_visual(self):
        self.meshcat_visual()

    def test_meshcat_collision(self):
        self.meshcat_collision()


class TestReachy_Meshcat_RightArm(BaseMeshcat):
    q = np.random.rand(7)
    wrapper = r2.pin.r_arm

    def test_meshcat_visual(self):
        self.meshcat_visual()

    def test_meshcat_collision(self):
        self.meshcat_collision()


class TestReachy_Meshcat_RightArm(BaseMeshcat):
    q = np.random.rand(3)
    wrapper = r2.pin.head

    def test_meshcat_visual(self):
        self.meshcat_visual()

    def test_meshcat_collision(self):
        self.meshcat_collision()
