import numpy as np
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!

rr.init("rerun_example_my_data", spawn=True)

positions = np.zeros((10, 3))
positions[:, 0] = np.linspace(-10, 10, 10)

colors = np.zeros((10, 3), dtype=np.uint8)
colors[:, 0] = np.linspace(0, 255, 10)

rr.log("my_points", rr.Points3D(positions, colors=colors, radii=0.5))


rr.init("rerun_example_my_data", spawn=True)

SIZE = 10

pos_grid = np.meshgrid(*[np.linspace(-10, 10, SIZE)] * 3)
positions = np.vstack([d.reshape(-1) for d in pos_grid]).T

col_grid = np.meshgrid(*[np.linspace(0, 255, SIZE)] * 3)
colors = np.vstack([c.reshape(-1) for c in col_grid]).astype(np.uint8).T

rr.log("my_points2", rr.Points3D(positions, colors=colors, radii=0.5))
