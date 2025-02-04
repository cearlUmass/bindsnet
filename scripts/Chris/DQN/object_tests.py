from Grid_Cells import Grid_Cell, GC_Module
import matplotlib.pyplot as plt
import numpy as np
NUM_CELLS = 4
x_offsets = [0]*4
y_offsets = [0]*4
rotations = [0]*4
scale = [1]*4
sharpness = [1]*4
gc_m = GC_Module(NUM_CELLS, x_offsets, y_offsets, rotations, scale, sharpness)
fig, ax = plt.subplots(figsize=(5, 5))
gs = fig.add_gridspec(1, 4)
for i, gc in enumerate(gc_m.grid_cells):
  ax = fig.add_subplot(gs[0, i])
  gc.plot_peaks([0,10], [0,10], pos=(0,0), contours=True, fig=fig, ax=ax)
# gc_m.grid_cells[0].plot_peaks([0,10], [0,10], pos=(5,5), contours=True, fig=fig, ax=ax)
# gc_m.plot_peaks([0,10], [0,10], pos=(5,5), contours=True, fig=fig, ax=ax)
print()
# gc.plot_peaks([0,10], [0,10], 'blue', pos=(5,5), contours=True)
# gc = Grid_Cell(x_offset=0.4, y_offset=0.1, rotation=.4, scale=1.6, sharpness=1.2,)
# gc.plot_peaks([0,10], [0,10], 'blue', pos=(5,5), contours=True)
# print()
# print(gc.generate((7.0, 3.3)))
# gc.plot_firing_peak()
# gc.plot_activity((7.0, 3.3), (0, 10), (0, 10))