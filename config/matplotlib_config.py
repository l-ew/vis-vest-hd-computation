import matplotlib.pyplot as plt
import seaborn as sns

TINY_SIZE = 5
SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 9
MARKER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('legend', title_fontsize=MEDIUM_SIZE)
plt.rc('lines', linewidth=1)
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['legend.frameon'] = False
plt.rcParams['lines.markersize'] = 9

plt.rc('font', family='sans-serif')
plt.rc('font', serif='Helvetica')
plt.rc('axes', linewidth=0.5)

DECODING_COLOR = 'tab:red'
HEAD_COLOR = 'tab:blue'
EYE_COLOR = 'tab:blue'

DARK_COLOR = (75/255, 75/255, 75/255)
NATURAL_COLOR = (147/255, 112/255, 219/255)
SYNC_COLOR = (255/255, 127/255, 14/255)
SCENE_COLOR = (44/255, 160/255, 44/255)
STIM_COLOR = (31/255, 119/255, 180/255)
SHADING_ALPHA = 0.5
NEUTRAL_COLOR = 'lightgrey'
NEUTRAL_EDGE_COLOR = 'dimgrey'
ACTIVITY_COLORMAP = 'Blues'

VEL_DASHES = {0: (),  # solid line, no dashes
              45: (4, 1.5),  # dash pattern with dashes of length 4 and gaps of length 1.5
              90: (1,1),  # dotted line with dashes and gaps of length 1
              135: (3, 1, 1, 1),  # dash-dot pattern with dashes of length 3, gaps of length 1, dots of length 1, and gaps of length 1
              180: (5, 2, 1, 2)}  # dash-dot-dot pattern with dashes of length 5, gaps of length 2, dots of length 1, and gaps of length 2

vels = [0, 45, 90, 135, 180]
VEL_PALETTE = sns.color_palette("cividis", n_colors=len(vels))[::-1]
VEL_PALETTE = dict(zip(vels, VEL_PALETTE))
