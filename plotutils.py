#  Copyright (c) 2022 Ludger Heide, Technische Universität Berlin.
#  All rights reserved pending open-source release.

import matplotlib

LINEWIDTH_IN_CM = 14.6979
TEXTHEIGHT_IN_CM = 20.93701
NORMAL_PLOT_HEIGHT = TEXTHEIGHT_IN_CM / (2.54 * 2.3)
FULLSIZE_PLOT_HEIGHT = (TEXTHEIGHT_IN_CM - 1) / 2.54
NORMAL_PLOT_WIDTH = LINEWIDTH_IN_CM / 2.54

pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": "Computer Modern",
    "text.usetex": True,
    "figure.figsize": (LINEWIDTH_IN_CM / 2.54, NORMAL_PLOT_HEIGHT),
    "axes.titlesize": 11,
    "lines.linewidth": 1.0,
}
# Define a colro scheme once I've figured it out
# 'axes.prop_cycle': cycler(color='bgrcmyk')
matplotlib.rcParams.update(pgf_with_rc_fonts)
