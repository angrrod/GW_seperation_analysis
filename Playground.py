
import gwpy  #visualization
import bilby
import numpy as np
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from bilby.gw.source import lal_binary_black_hole

#TODO: test for multiple detectors

import lalsimulation
import os,inspect
# Path to the installed lalsimulation Python module
path = inspect.getfile(lalsimulation)
print(path)

# Navigate upward to see the package tree
print(os.path.dirname(path))