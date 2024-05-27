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
    import pytest

    import reachy2_modelling as r2

except ImportError as e:
    print("Error:", e)
    print("is reachy2_modelling installed? run in root directory:")
    print("pip install -e .")
    exit(1)

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


class BaseFK:
    def kdl_vs_pin_torsoframe(self):
        kdl = self.kdl_torso
        assert np.allclose(
            kdl.fk_as_SE3(self.q),
            self.pin.fk(self.q, tip=self.tip, torso_frame=not kdl.is_world_frame()),
        )

    def kdl_vs_pin_worldframe(self):
        kdl = self.kdl_world
        assert np.allclose(
            kdl.fk_as_SE3(self.q),
            self.pin.fk(self.q, tip=self.tip, torso_frame=not kdl.is_world_frame()),
        )

    def kdl_vs_rtb_torsoframe(self):
        kdl = self.kdl_torso
        assert np.allclose(
            kdl.fk_as_SE3(self.q),
            self.rtb.fk_as_SE3(self.q, tip=self.tip, world_frame=kdl.is_world_frame()),
        )

    def kdl_vs_rtb_worldframe(self):
        kdl = self.kdl_world
        assert np.allclose(
            kdl.fk_as_SE3(self.q),
            self.rtb.fk_as_SE3(self.q, tip=self.tip, world_frame=kdl.is_world_frame()),
        )

    def pin_vs_rtb_torsoframe(self):
        torsoframe = True
        assert np.allclose(
            self.pin.fk(self.q, tip=self.tip, torso_frame=torsoframe),
            self.rtb.fk_as_SE3(self.q, tip=self.tip, world_frame=not torsoframe),
        )

    def pin_vs_rtb_worldframe(self):
        torsoframe = False
        assert np.allclose(
            self.pin.fk(self.q, tip=self.tip, torso_frame=torsoframe),
            self.rtb.fk_as_SE3(self.q, tip=self.tip, world_frame=not torsoframe),
        )


class TestReachyFK_LeftArm(BaseFK):
    q = np.random.rand(7)
    tip = r2.constants.l_arm_tip
    kdl_world = r2.old.kdl.l_arm_world
    kdl_torso = r2.old.kdl.l_arm_torso
    pin = r2.old.pin.l_arm
    rtb = r2.old.rtb.l_arm

    def test_kdl_vs_pin_torsoframe(self):
        self.kdl_vs_pin_torsoframe()

    def test_kdl_vs_pin_worldframe(self):
        self.kdl_vs_pin_worldframe()

    @pytest.mark.skip(reason="rtb seems to not be able to change tip easily")
    def test_kdl_vs_rtb_torsoframe(self):
        self.kdl_vs_rtb_torsoframe()

    def test_kdl_vs_rtb_worldframe(self):
        self.kdl_vs_rtb_worldframe()

    @pytest.mark.skip(reason="rtb seems to not be able to change tip easily")
    def test_pin_vs_rtb_torsoframe(self):
        self.pin_vs_rtb_torsoframe()

    def test_pin_vs_rtb_worldframe(self):
        self.pin_vs_rtb_worldframe()


class TestReachyFK_RightArm(BaseFK):
    q = np.random.rand(7)
    tip = r2.constants.r_arm_tip
    kdl_world = r2.old.kdl.r_arm_world
    kdl_torso = r2.old.kdl.r_arm_torso
    pin = r2.old.pin.r_arm
    rtb = r2.old.rtb.r_arm

    def test_kdl_vs_pin_torsoframe(self):
        self.kdl_vs_pin_torsoframe()

    def test_kdl_vs_pin_worldframe(self):
        self.kdl_vs_pin_worldframe()

    @pytest.mark.skip(reason="rtb seems to not be able to change tip easily")
    def test_kdl_vs_rtb_torsoframe(self):
        self.kdl_vs_rtb_torsoframe()

    def test_kdl_vs_rtb_worldframe(self):
        self.kdl_vs_rtb_worldframe()

    @pytest.mark.skip(reason="rtb seems to not be able to change tip easily")
    def test_pin_vs_rtb_torsoframe(self):
        self.pin_vs_rtb_torsoframe()

    def test_pin_vs_rtb_worldframe(self):
        self.pin_vs_rtb_worldframe()


class TestReachyFK_Head(BaseFK):
    q = np.random.rand(3)
    tip = r2.constants.head_tip
    kdl_world = r2.old.kdl.head_world
    kdl_torso = r2.old.kdl.head_torso
    pin = r2.old.pin.head
    rtb = r2.old.rtb.head

    def test_kdl_vs_pin_torsoframe(self):
        self.kdl_vs_pin_torsoframe()

    def test_kdl_vs_pin_worldframe(self):
        self.kdl_vs_pin_worldframe()

    @pytest.mark.skip(reason="rtb seems to not be able to change tip easily")
    def test_kdl_vs_rtb_torsoframe(self):
        self.kdl_vs_rtb_torsoframe()

    def test_kdl_vs_rtb_worldframe(self):
        self.kdl_vs_rtb_worldframe()

    @pytest.mark.skip(reason="rtb seems to not be able to change tip easily")
    def test_pin_vs_rtb_torsoframe(self):
        self.pin_vs_rtb_torsoframe()

    def test_pin_vs_rtb_worldframe(self):
        self.pin_vs_rtb_worldframe()


class TestReachyFK_Examples(BaseFK):
    def test_kdl_head(self):
        q = np.random.rand(3)
        kdl = r2.old.kdl.head_world
        M = kdl.fk_as_SE3(q)
        kdl = r2.old.kdl.head_torso
        M = kdl.fk_as_SE3(q)

    def test_kdl_arms(self):
        q = np.random.rand(7)
        kdl = r2.old.kdl.l_arm_world
        M = kdl.fk_as_SE3(q)
        kdl = r2.old.kdl.l_arm_torso
        M = kdl.fk_as_SE3(q)

        kdl = r2.old.kdl.r_arm_world
        M = kdl.fk_as_SE3(q)
        kdl = r2.old.kdl.r_arm_torso
        M = kdl.fk_as_SE3(q)

    def test_rtb_head(self):
        q = np.random.rand(3)
        rtb = r2.old.rtb.head
        tip = r2.constants.head_tip
        rtb.fk_as_SE3(q, tip=tip, world_frame=False)
        rtb.fk_as_SE3(q, tip=tip, world_frame=True)

    def test_rtb_arms(self):
        q = np.random.rand(3)
        rtb = r2.old.rtb.l_arm
        tip = r2.constants.l_arm_tip
        M = rtb.fk_as_SE3(q, tip=tip, world_frame=False)
        M = rtb.fk_as_SE3(q, tip=tip, world_frame=True)

        rtb = r2.old.rtb.r_arm
        tip = r2.constants.r_arm_tip
        M = rtb.fk_as_SE3(q, tip=tip, world_frame=False)
        M = rtb.fk_as_SE3(q, tip=tip, world_frame=True)

    def test_pin_head(self):
        q = np.random.rand(3)
        pin = r2.old.pin.head
        tip = r2.constants.head_tip
        M = pin.fk(q, tip=tip, torso_frame=False)
        M = pin.fk(q, tip=tip, torso_frame=True)

    def test_pin_arms(self):
        q = np.random.rand(7)
        pin = r2.old.pin.l_arm
        tip = r2.constants.l_arm_tip
        M = pin.fk(q, tip=tip, torso_frame=False)
        M = pin.fk(q, tip=tip, torso_frame=True)

        pin = r2.old.pin.r_arm
        tip = r2.constants.r_arm_tip
        M = pin.fk(q, tip=tip, torso_frame=False)
        M = pin.fk(q, tip=tip, torso_frame=True)
