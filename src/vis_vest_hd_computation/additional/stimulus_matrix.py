import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from vis_vest_hd_computation.imports import *


def plot_stimulus_matrix(m, c, n, vels, fname='stimulus_matrix.eps', fig_path='.'):
    fig = plt.figure(figsize=(2.35, 2.0))
    ax = fig.add_axes([0.25, 0.3, 0.55, 0.6])
    ax.matshow(m, cmap=c, norm=n, extent=[-202.5,202.5,-202.5,202.5])
    ax.set(aspect='equal')
    ticks = list(vels-22.5) + [vels[-1] + 22.5]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(x) for x in list(vels)] + [''])
    ax.set_xticklabels([str(x) for x in list(vels)] + [''])
    ax.grid(c='k', ls='-', lw='0.5')
    ax.tick_params('both', length=0, which='both')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylabel('Peak {} velocity\n [deg/s]'.format(PLATFORM_STR.lower()))
    ax.set_xlabel('Peak {} velocity\n [deg/s]'.format(SCENE_STR.lower().replace('_', ' ')))
    plt.setp(ax.get_yticklabels(), va="bottom")
    ax.spines[['right', 'top']].set_visible(True)
    ax.tick_params(axis='x', which='minor', pad=2)
    ax.set_xticks(vels, minor=True)
    ax.set_xticklabels('')
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
    plt.setp(ax.get_xticklabels(minor=True), rotation=45, va="top", ha='right', rotation_mode="anchor")

    colors = sns.color_palette("GnBu_d", n_colors=5)
    colors = colors[1:]
    colors = colors[::-1] + ['w'] + colors
    for vel, c in zip(vels, colors):
        if vel > 0:
            ax.annotate("", xy=(202.05, vel), xytext=(260, vel), arrowprops=dict(arrowstyle="->", color=c, lw=0.5))
        elif vel < 0:
            ax.annotate("", xy=(202.05, vel), xytext=(260, vel), arrowprops=dict(arrowstyle="->", color=c, lw=0.5))

    fig.savefig(os.path.join(fig_path, fname), dpi=300, format='eps')
    plt.close(fig)


def main():

    vels = 45 * np.arange(-4,5)
    n_vels = len(vels)
    m = np.zeros((n_vels, n_vels), dtype=int)

    np.fill_diagonal(m, 2)
    m = m[::-1,:]

    m[n_vels//2,:] = 3
    m[:,n_vels//2] = 1
    m[n_vels//2, n_vels//2] = 4

    lightgrey = (0.9, 0.9, 0.9)
    c = mpl.colors.ListedColormap([lightgrey, NATURAL_COLOR, SYNC_COLOR, SCENE_COLOR, lightgrey])
    n = mpl.colors.Normalize(vmin=0,vmax=4)

    fig_path = os.path.join(BASE_PATH, 'figures', 'stimulus_matrix')
    os.makedirs(fig_path, exist_ok=True)

    plot_stimulus_matrix(m, c, n, vels, fig_path=fig_path)


if __name__ == "__main__":
    main()
