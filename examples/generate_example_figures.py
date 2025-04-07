import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import cmocean
from scipy import ndimage
from vis_vest_hd_computation.imports import *
from vis_vest_hd_computation.utils import comp_delta, rayleigh_vector, map_angle, comp_tuning_curves, comp_peak_firing_direction
from vis_vest_hd_computation.spectral_embedding import plot_embedding_colored, align_decoded_direction_with_hd
import vis_vest_hd_computation.visualization as visualization
from vis_vest_hd_computation.analyze_recording import start_stop_inds


def plot_embedding_two_stages(embedding, mask, colors, fig_path, axis_off=True, high=0.04, fig_name='embedding.pdf'):
    factor = 1.25
    xmin = factor * np.percentile(embedding[:,0], 1)
    xmax = factor * np.percentile(embedding[:,0], 99)
    ymin = factor * np.percentile(embedding[:,1], 1)
    ymax = factor * np.percentile(embedding[:,1], 99)
    axmin = np.minimum(xmin, ymin)
    axmax = np.maximum(xmax, ymax)

    plot_idx = np.random.permutation(len(mask))

    cmap = mcol.ListedColormap([(colors[0], 0.5), (colors[1], 0.5)])
    bounds = [0, 0.5, 1]
    norm = mcol.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    im = ax.scatter(embedding[plot_idx,0], embedding[plot_idx,1], c=mask[plot_idx], s=0.5, cmap=cmap, norm=norm)

    ax.set_xlim([axmin, axmax])
    ax.set_ylim([axmin, axmax])
    ax.set_aspect('equal')

    if axis_off:
        plt.axis('off')
    else:
        ax.tick_params(axis='both', which='both', bottom=False, left=False, top=False, labelleft=False, labelbottom=False)
        #ax.set_xlabel('Embedding dimension 1')
        #ax.set_ylabel('Embedding dimension 2')

        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)

        ax.spines['left'].set_bounds(low=0, high=high)
        ax.spines['bottom'].set_bounds(low=0, high=high)

        # Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
        # case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
        # respectively) and the other one (1) is an axes coordinate (i.e., at the very
        # right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
        # actually spills out of the axes.
        ax.plot(-axmin / (-axmin + axmax) + high / (-axmin + axmax), 0, ">k", markersize=3, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, -axmin / (-axmin + axmax) + high / (-axmin + axmax), "^k", markersize=3, transform=ax.get_xaxis_transform(), clip_on=False)

    fig.savefig(os.path.join(fig_path, '{}.pdf'.format(fig_name)), bbox_inches='tight')
    plt.close()
    return axmin, axmax


def plot_embedding(embedding, fig_path, fig_name='embedding.pdf'):
    factor = 1.25
    xmin = factor * np.percentile(embedding[:,0], 1)
    xmax = factor * np.percentile(embedding[:,0], 99)
    ymin = factor * np.percentile(embedding[:,1], 1)
    ymax = factor * np.percentile(embedding[:,1], 99)
    axmin = np.minimum(xmin, ymin)
    axmax = np.maximum(xmax, ymax)

    plot_idx = np.random.permutation(embedding.shape[0])

    fig, ax = plt.subplots(figsize=(2, 2))
    im = ax.scatter(embedding[plot_idx,0], embedding[plot_idx,1], c='k', alpha=0.5, s=0.5);
    #im = ax.scatter(embedding[:,0], embedding[:,1], c=passive, s=0.5, cmap=cmap, norm=norm);

    ax.set_xlim([axmin, axmax])
    ax.set_ylim([axmin, axmax])
    ax.set_aspect('equal');
    plt.axis('off')

    fig.savefig(os.path.join(fig_path, '{}.pdf'.format(fig_name)), bbox_inches='tight')
    plt.close()
    return axmin, axmax


def export_legend(legend, fig_path, filename="legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(fig_path, filename), bbox_inches=bbox)


def active_passive_legend(colors, fig_path):
    f = lambda c: plt.scatter([],[], color=c)
    handles = [f(c) for c in colors]
    labels = ['Active + landmarks', 'Passive + grating']
    legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)
    export_legend(legend, fig_path, filename="active_passive_legend.pdf")
    plt.close()


def overlay_tuning_plots(tuning1, tuning2, theta, vec_length1, vec_length2, c1='k', c2='r', fname='overlay_tuning.pdf', fig_path='.'):

    theta = np.hstack([theta, theta[0]])

    fig, axs = plt.subplots(7, 15, subplot_kw={'projection': 'polar'}, figsize=[7.0, 3.5])

    for k, ax in enumerate(axs.reshape(-1)):

        if k < len(vec_length1):

            f1 = tuning1[k,:]
            f1 = np.hstack([f1, f1[0]])

            f2 = tuning2[k,:]
            f2 = np.hstack([f2, f2[0]])

            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])

            plt.text(0.2, 1.0, '{:.2f}'.format(vec_length1[k]), ha='center', va='bottom', color=c1, transform = ax.transAxes, fontsize=5)
            plt.text(0.8, 1.0, '{:.2f}'.format(vec_length2[k]), ha='center', va='bottom', color=c2, transform = ax.transAxes, fontsize=5)

            ax.spines['polar'].set_visible(True)
            ax.spines['polar'].set_linewidth(0.5)
            ax.spines['polar'].set_edgecolor('darkgrey')

            ax.plot(theta, f1, color= c1, alpha= 1.0, linewidth=0.5, zorder=4, clip_on=False)
            ax.fill_between(x=theta, y1= f2, color= c2, alpha= 1.0, zorder=3, clip_on=False)
            ax.set_rmax(1)
            ax.set_rticks([0.5, 1])
            ax.grid(linewidth=0.5)

        else:
            ax.axis('off')

    plt.subplots_adjust(wspace=None, hspace=None)
    fig.savefig(os.path.join(fig_path, fname), bbox_inches='tight')
    plt.close(fig)


def tuning_heatmap(tuning_curves, aspect=0.75, ylabel='internal direction', path='tuning_heatmap.pdf'):

    fig, ax = plt.subplots(figsize=[1.25, 1.75])
    im = ax.imshow(tuning_curves, cmap=ACTIVITY_COLORMAP, vmin=0, vmax=1, aspect=aspect, interpolation='none')
    ax.set_ylabel('Neuron # (sorted)');
    ax.set_xticks(np.linspace(-0.5, 35.5, 3))
    ax.set_xticklabels([0, 180, 360])
    ax.set_xlabel('{} [deg]'.format(ylabel))
    ax.xaxis.set_ticks_position('bottom')

    fig.subplots_adjust(right=0.7)
    cbar_ax = fig.add_axes([0.725, 0.3, 0.03, 0.4])
    fig.colorbar(im, cax=cbar_ax, label = 'Normalized activity [a.u.]', ticks=[0, 0.5, 1])

    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def tuning_plots(tuning_active, theta, vec_length, example_neurons, fig_path='.'):

    theta = np.hstack([theta, theta[0]])

    fig, axs = plt.subplots(3, 2, subplot_kw={'projection': 'polar'}, figsize=[0.9, 1.6])

    for k, ax in zip(example_neurons, axs.reshape(-1)):

        f = tuning_active[k,:]
        f = np.hstack([f, f[0]])

        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

        plt.text(0.5, 1.0, 'r = {:.2f}'.format(vec_length[k]), ha='center', va='bottom', transform=ax.transAxes, fontsize=TINY_SIZE)

        ax.spines['polar'].set_visible(True)
        ax.spines['polar'].set_linewidth(0.5)
        ax.spines['polar'].set_edgecolor('darkgrey')

        ax.fill_between(x=theta, y1= f, color= "k", alpha= 1, zorder=3, clip_on=False)
        ax.set_rmax(1)
        ax.set_rticks([0.5, 1])
        ax.grid(linewidth=0.5)

    plt.subplots_adjust(wspace=None, hspace=None)
    fig.savefig(os.path.join(fig_path, 'example_tuning_curves.pdf'), bbox_inches='tight')
    plt.close(fig)


def draw_brace(ax, xspan, yy, text, fontsize):
    """Draws an annotated brace on the axes."""
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1, clip_on=False, zorder=10)

    ax.text((xmax+xmin)/2., yy+.07*yspan, text, ha='center', va='bottom', fontsize=fontsize)


def plot_circ_traces(ax, t, hd, decoded_direction, t0=0, adjust=False, neg_angle=False):

    if adjust:
        t_hd = t + DECODED_PHYSICAL_LAG
    else:
        t_hd = t

    visualization.plot_circ_trace(ax, t_hd, hd, c=HEAD_COLOR, label='Head')

    visualization.plot_circ_trace(ax, t, decoded_direction, c=DECODING_COLOR, label='Decoding')

    if neg_angle:
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels([-180, 0, 180]);
    else:
        ax.set_ylim(0, 2*np.pi)
        ax.set_yticks([0, np.pi, 2*np.pi])
        ax.set_yticklabels([0, 180, 360]);

    ax.set_ylabel('Direction [deg]')

    mins = (t[-1] - t[0]) // 60
    sec = int(t[-1] - t[0])

    if mins > 0:
        ax.set_xlim(0, mins*60)
        ax.set_xticks(60 * np.arange(mins+1))
        ax.set_xticklabels(int(t[0]) + 60 * np.arange(mins+1, dtype=int))
    elif sec > 10:
        ax.set_xticks(10 * np.arange(int(np.ceil(t[0]/10)), t[-1]//10+1))
    else:
        ax.set_xticks([0, 3])
    ax.set_xlabel('t [s]')


def plot_population_act(C, t, hd, decoded_direction, peak_firing_dir=None, fname='pop_act.pdf', adjust=False, fps=12, fig_path='.', xlim=None, show_legend=True, t_start=None, t_stop=None, trial_numbers=None, vmax=1):

    fig, axs = plt.subplots(2, 1, figsize=[3.0, 2.25], sharex=True, gridspec_kw=dict(height_ratios=[1,1], hspace=0.25))
    fig.subplots_adjust(left=0.175, right=0.8, top=0.925, bottom=0.175)

    im = axs[0].imshow(C, extent=[t[0],t[-1],0,1], aspect='auto', cmap=ACTIVITY_COLORMAP, origin='lower', interpolation='none', vmin=0, vmax=vmax)
    axs[0].set_ylabel('Neuron #\n(sorted)', labelpad=5)

    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].tick_params(axis=u'both', which=u'both', length=0)

    plot_circ_traces(axs[1], t, hd, decoded_direction, adjust=adjust)
    if xlim is not None:
        axs[0].set_xlim(xlim)
        axs[1].set_xlim(xlim)

    if adjust:
        adjustment = DECODED_PHYSICAL_LAG
    else:
        adjustment = 0

    if t_start is not None:
        for trial_number, ti, tf in zip(trial_numbers, t_start, t_stop):
            axs[1].vlines(ti+adjustment, 0, 2*np.pi, color=(65/255,65/255,65/255), linestyle='--', linewidth=1, zorder=-1)
            if trial_number == 79:
                draw_brace(axs[1], [ti+adjustment, tf+adjustment], 2 * np.pi + 0.25, 'trial #{}'.format(trial_number), fontsize=TINY_SIZE)
            else:
                draw_brace(axs[1], [ti+adjustment, tf+adjustment], 2 * np.pi + 0.25, '#{}'.format(trial_number), fontsize=TINY_SIZE)

            axs[1].vlines(tf+adjustment, 0, 2*np.pi, color=(65/255,65/255,65/255), linestyle='--', linewidth=1, zorder=-1)

    cbar_ax = fig.add_axes([0.825, 0.625, 0.015, 0.25])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Norm. activity [a.u.]', ticks=[0, vmax])
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=SMALL_SIZE)

    peak_dir_name = 'Pref. head direction'

    if peak_firing_dir is not None:
        ax_pos = axs[0].get_position()
        pfd_ax = fig.add_axes([0, ax_pos.y0, 0.1, ax_pos.height])
        pfd_ax.imshow(peak_firing_dir.reshape(len(peak_firing_dir),1), aspect=0.25, cmap=cmocean.cm.phase, origin='lower', interpolation='none', vmin=0, vmax=2*np.pi)
        pfd_ax.axis('off')
        plt.text(0.0, 0.5, peak_dir_name, ha='right', va='center', transform = pfd_ax.transAxes, rotation=90, fontsize=SMALL_SIZE)

    if show_legend:
        axs[1].legend(loc=(0,1), mode="expand", ncol=2, fontsize=SMALL_SIZE)

    axs[0].spines['right'].set_visible(True)
    axs[0].spines['top'].set_visible(True)
    axs[1].spines['right'].set_visible(True)
    axs[1].spines['top'].set_visible(True)

    fig.savefig(os.path.join(fig_path, fname), dpi=300)
    plt.close()


def filter_trace(x):
    filtered_cos = ndimage.median_filter(np.cos(x), size=80)
    filtered_sin = ndimage.median_filter(np.sin(x), size=80)
    filtered = np.arctan2(filtered_sin, filtered_cos)
    filtered[np.hstack((np.abs(np.diff(filtered)) > np.pi, False))] = np.nan
    return filtered


def extract_active_passive(x, passive, passive_start, passive_end, add_nan=False):
    x_passive = x[passive][passive_start:passive_end]
    x_active = x[~passive]
    if add_nan:
        x_active[-1] = np.nan
    x_active_passive = np.hstack((x_active, x_passive))
    return x_active_passive


def plot_directions_and_drift(t, hd, decoded_direction, delta, passive, passive_start, passive_end, fps=12, fig_path='.'):

    t_passive = t[passive][passive_start:passive_end] - t[passive][passive_start]
    t_passive_start = t[~passive][-1]
    t_active_passive = np.hstack((t[~passive], t_passive + t_passive_start))

    hd_active_passive = extract_active_passive(hd, passive, passive_start, passive_end, add_nan=True)
    decoded_direction_active_passive = extract_active_passive(decoded_direction, passive, passive_start, passive_end, add_nan=True)
    delta_active_passive = extract_active_passive(delta, passive, passive_start, passive_end)

    fig, axs = plt.subplots(2, 1, figsize=[6.75, 2], sharex=True, gridspec_kw=dict(height_ratios=[1,1], wspace=0))

    visualization.plot_circ_trace(axs[0], t_active_passive, hd_active_passive, c=HEAD_COLOR, label='Head direction', linewidth=0.5)
    visualization.plot_circ_trace(axs[0], t_active_passive, decoded_direction_active_passive, c=DECODING_COLOR, label='Decoded direction', linewidth=0.5)
    axs[0].set_ylabel('Direction [deg]')
    axs[0].set_ylim(0, 2*np.pi)
    axs[0].set_yticks([0, np.pi, 2*np.pi])
    axs[0].set_yticklabels([0, 180, 360])
    axs[0].vlines(t_passive_start, ymin=0, ymax=2*np.pi, color='k', linestyle='--', lw=0.75)

    axs[1].scatter(t_active_passive, delta_active_passive, c='k', s=0.25, alpha=0.25)
    #axs[1].plot(t_active_passive, filtered_delta_active_passive, 'k', lw=1)
    axs[1].set_ylabel('Drift [deg]')
    axs[1].set_ylim(-np.pi, np.pi)
    axs[1].set_yticks([-np.pi, 0, np.pi])
    axs[1].set_yticklabels([-180, 0, 180]);
    axs[1].vlines(t_passive_start, ymin=-np.pi, ymax=np.pi, color='k', linestyle='--', lw=0.75)
    #axs[1].axhline(0, color='r', linestyle='--', lw=0.75)

    for ax in axs:
        ax.set_xlim(0, t_active_passive[-1])
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)

    step = 5 * 60
    steps_active = t_passive_start // step
    ticks_active = step * np.arange(steps_active+1)
    ticklabels_active = 5 * np.arange(steps_active+1, dtype='int')

    steps_passive = (t_active_passive[-1] - t_passive_start) // step
    ticks_passive = t_passive_start + step * np.arange(steps_passive+1)
    ticklabels_passive = 5 * np.arange(steps_passive+1, dtype='int')

    axs[1].set_xticks(np.hstack((ticks_active, ticks_passive)))
    axs[1].set_xticklabels(np.hstack((ticklabels_active, ticklabels_passive)))

    axs[1].set_xlabel('t [min]')

    plt.savefig(os.path.join(fig_path, 'drift_over_time.pdf'), bbox_inches='tight')
    plt.close()


def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """

    if invalid is None: invalid = np.isnan(data)
    ind = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]


def get_theo_trajectory(v, t, T=3):
    x = v * T * (np.cos(np.pi * (t - t[0]) / T) - 1) / 180
    return x


def get_total_rot(v, T=3):
    x = -2 * v * T / 180
    return x


def frmd7_active_example(session):

    fig_path = os.path.join(BASE_PATH, 'examples', session)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    with open(os.path.join(BASE_PATH, 'data', 'embedding_data', '{}.p'.format(session)), 'rb') as file:
        data = pickle.load(file)

    embedding = data['embedding']
    hd = data['hd']
    hd = map_angle(hd)

    plot_embedding_colored(embedding, hd, -0.1, 0.1, 0, fig_path, axis_off=False, vmin=0, vmax=2*np.pi)
    visualization.cmap_legend(fig_path)

    hd_bins = 36
    hd_bin_edges = np.linspace(0, 2*np.pi, hd_bins + 1)
    tuning, hd_bin_centers = comp_tuning_curves(hd, data['deconvolved_act'], hd_bin_edges)
    pref_dir, vec_length = rayleigh_vector(tuning, hd_bin_centers)
    pref_dir = map_angle(pref_dir)

    peak_firing_dir, ordering = comp_peak_firing_direction(tuning, hd_bin_centers)
    peak_firing_dir = peak_firing_dir[ordering]
    tuning = tuning[ordering,:]
    vec_length = vec_length[ordering]

    tuning_heatmap(tuning, aspect=1.4, ylabel='Head direction', path=os.path.join(fig_path, 'hd_tuning_heatmap.pdf'))

    example_neurons = [5, 16, 24, 28, 34, 42]
    tuning_plots(tuning, hd_bin_centers, vec_length, example_neurons, fig_path=fig_path)

    C = data['deconvolved_act'][:,ordering].T
    t = data['time']
    C = norm_act(C, norm_act=C)
    decoded_direction = data[DECODED_DIRECTION]
    decoded_direction = map_angle(decoded_direction)
    fps = 12
    plt_range = np.arange(0*60*fps, 2*60*fps+1)

    vmax = 0.4
    fname = 'population_activity_active.pdf'
    plot_population_act(C[:,plt_range], t[plt_range], hd[plt_range], decoded_direction[plt_range], peak_firing_dir, fname, vmax=vmax, fig_path=fig_path)


def comp_hd(t, start_inds, stop_inds, motor_directions, motor_vels):
    hd = np.nan * np.zeros(len(t))
    x = 0
    v_max = [0, 45, 90, 135, 180]
    for start_ind, stop_ind, d, k in zip(start_inds, stop_inds, motor_directions, motor_vels):
        v = d * v_max[k]
        hd[start_ind:stop_ind] = x + get_theo_trajectory(v, t[start_ind:stop_ind])
        if v_max[k] != 0:
            total_rot = get_total_rot(v)
            x += total_rot
    return hd


def plot_platform_velocity(T=3, fig_name='platform_velocity.pdf', fig_path='.'):

    plt.figure(figsize=(2.5,1.75))
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.35, bottom=0.3, right=0.9, top=0.9, wspace=0, hspace=0)

    v_max = [0, 45, 90, 135, 180]
    t = np.linspace(0,T,3001)
    for v in v_max:
        y = v * np.sin(np.pi * t / T)
        plt.plot(t, y, color=STIM_COLOR, zorder=3, clip_on=False)
    plt.xticks(np.arange(T+1))
    plt.xlim(0,T)
    plt.yticks(v_max)
    plt.ylim(0, v_max[-1])
    plt.xlabel('t [s]')
    plt.ylabel('{} velocity\n[deg/s]'.format(PLATFORM_STR).capitalize())

    plt.savefig(os.path.join(fig_path, "platform_velocity_over_time.pdf"))
    plt.close()


def norm_act(act, norm_act=None, use_perc=False, perc=99.5):
    if norm_act is None:
        norm_act = act

    if use_perc:
        return act / np.percentile(norm_act, perc, axis=1)[:, np.newaxis]
    else:
        return act / np.max(norm_act, axis=1)[:, np.newaxis]


def combined_example(session, example_neurons=[], fps=FPS_MINISCOPE, combined=False):

    fig_path = os.path.join(BASE_PATH, 'examples', session)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    with open(os.path.join(BASE_PATH, 'data', 'embedding_data', '{}.p'.format(session)), 'rb') as file:
        data = pickle.load(file)

    embedding = data['embedding']
    stage = data['stage']
    passive = stage == 1

    active_color = '#fce166'  # '#fce166'  #'#b9a281'
    passive_color = '#c79fef'  # '#95d0fc'  #'#bf77f6' #"#607c8e"
    active_passive_colors = [active_color, passive_color]

    fig_name = 'embedding_active_passive.pdf'
    axmin, axmax = plot_embedding_two_stages(embedding, passive, active_passive_colors, fig_path, axis_off=False, high=0.025, fig_name=fig_name)
    active_passive_legend(active_passive_colors, fig_path)

    t = data['time']
    hd = data['hd']
    hd = map_angle(hd)

    fig_name = 'embedding_HD.pdf'
    plot_embedding_colored(embedding[~passive], hd[~passive], axmin, axmax, 0, fig_path, axis_off=False, high=0.025, vmin=0, vmax=2*np.pi)
    visualization.cmap_legend(fig_path)

    act = data['deconvolved_act']
    decoded_direction = data[DECODED_DIRECTION]
    drift = comp_delta(decoded_direction, hd)
    decoded_direction, drift = align_decoded_direction_with_hd(decoded_direction, drift, ~passive)
    decoded_direction = map_angle(decoded_direction)

    act_active = act[~passive]
    hd_active = hd[~passive]

    hd_bins = 36
    hd_bin_edges = np.linspace(0, 2*np.pi, hd_bins + 1)
    tuning_active, hd_bin_centers = comp_tuning_curves(hd_active, act_active, hd_bin_edges)
    pref_dir, vec_length = rayleigh_vector(tuning_active, hd_bin_centers)
    pref_dir = map_angle(pref_dir)

    peak_firing_dir, ordering = comp_peak_firing_direction(tuning_active, hd_bin_centers)
    peak_firing_dir = peak_firing_dir[ordering]
    tuning_active = tuning_active[ordering,:]
    vec_length = vec_length[ordering]

    tuning_heatmap(tuning_active, ylabel='Head direction', path=os.path.join(fig_path, 'hd_tuning_heatmap_active.pdf'))

    #example_neurons = [2, 14, 33, 53, 63, 82]
    #tuning_plots(tuning_active, hd_bin_centers, vec_length, example_neurons, fig_path=fig_path)

    mins = 0.5
    n_samples = int(60*mins*fps) + 2
    k0 = 320 * fps  #0 #320 
    plt_range = np.arange(k0, k0+n_samples)
    C = act[k0:k0+n_samples,ordering].T 
    C = norm_act(C, norm_act=act_active.T[ordering,:])
    vmax = 0.4

    fname = 'population_activity_active.pdf'
    plot_population_act(C, t[plt_range]-t[plt_range][0], hd[plt_range], decoded_direction[plt_range], peak_firing_dir, fname, vmax=vmax, fig_path=fig_path) #aspect=10

    decoded_direction_active = decoded_direction[~passive]
    decoded_direction_passive = decoded_direction[passive]
    act_passive = act[passive]

    latent_tuning_active, hd_bin_centers = comp_tuning_curves(decoded_direction_active, act_active, hd_bin_edges)
    latent_tuning_passive, hd_bin_centers = comp_tuning_curves(decoded_direction_passive, act_passive, hd_bin_edges)

    latent_tuning_active = latent_tuning_active[ordering,:]
    latent_tuning_passive = latent_tuning_passive[ordering,:]

    tuning_heatmap(latent_tuning_active, ylabel='Decoded HD', path=os.path.join(fig_path, 'latent_tuning_heatmap_active.pdf'))
    tuning_heatmap(latent_tuning_passive, ylabel='Decoded HD', path=os.path.join(fig_path, 'latent_tuning_heatmap_passive.pdf'))

    tuning_passive, hd_bin_centers = comp_tuning_curves(hd[passive], act_passive, hd_bin_edges)
    tuning_passive = tuning_passive[ordering,:]
    tuning_heatmap(tuning_passive, ylabel='Head direction', path=os.path.join(fig_path, 'hd_tuning_heatmap_passive.pdf'))

    _, latent_vec_length_passive = rayleigh_vector(latent_tuning_passive, hd_bin_centers)
    _, latent_vec_length_active = rayleigh_vector(latent_tuning_active, hd_bin_centers)

    fname = 'latent_tuning_active_vs_passive.pdf'
    overlay_tuning_plots(latent_tuning_active, latent_tuning_passive, hd_bin_centers, latent_vec_length_active, latent_vec_length_passive, c1='k', c2=passive_color, fname=fname, fig_path=fig_path)

    fname = 'hd_vs_latent_latent_tuning_active.pdf'
    overlay_tuning_plots(latent_tuning_active, tuning_active, hd_bin_centers, latent_vec_length_active, vec_length, c1='k', c2=HEAD_COLOR, fname=fname, fig_path=fig_path)

    passive_start = np.sum(~passive) + np.where(data['time'][passive] > data['stimulus_start'][0])[0][0]
    act_passive = act[passive_start:,ordering]

    stim_path = os.path.join(BASE_PATH, 'data', 'stim_sets', session)
    with np.load(os.path.join(stim_path, 'motor_directions.npz')) as npz:
        motor_directions = np.ma.MaskedArray(**npz)
    motor_speeds = np.load(os.path.join(stim_path,'motor_speeds.npy'))

    hd_theo = np.nan * np.zeros(act_passive.shape[0])
    x = 0
    t_passive = data['time'][passive_start:]
    start_inds, stop_inds = start_stop_inds(t_passive, data['stimulus_start'], data['stimulus_stop'])

    v_max = [0, 45, 90, 135, 180]
    for start_ind, stop_ind, d, k in zip(start_inds, stop_inds, motor_directions, motor_speeds):
        v = d * v_max[k]
        hd_theo[start_ind:stop_ind] = x + get_theo_trajectory(v, t_passive[start_ind:stop_ind])
        if v_max[k] != 0:
            total_rot = get_total_rot(v)
            x += total_rot
            hd_theo[stop_ind] = hd_theo[start_ind] + total_rot

    tspan = 30
    n_samples = tspan * fps + 2
    start_trial = 78
    t_start = 444
    plt_start = np.where(t_passive > t_start)[0][0]
    plt_range = np.arange(plt_start, plt_start+n_samples)

    C = act_passive[plt_start:plt_start+n_samples,:].T 
    C = norm_act(C, norm_act=act_passive.T)

    decoded_direction_passive = decoded_direction[passive_start:]
    hd_theo = fill(hd_theo)
    offset = np.angle(np.mean(np.exp(1j * (decoded_direction_passive[plt_range] - hd_theo[plt_range]))))
    hd_theo = hd_theo + offset

    hd_theo = np.angle(np.exp(1j * hd_theo))
    hd_theo = map_angle(hd_theo)

    n_trials = 6
    trial_numbers = np.arange(start_trial+1, start_trial+n_trials+1)

    t_start = data['stimulus_start'][start_trial:start_trial+n_trials]
    t_stop = data['stimulus_stop'][start_trial:start_trial+n_trials]

    shift_t0 = True
    if shift_t0:
        t0 = t_passive[plt_range][0]
        t_start = [x - t0 for x in t_start]
        t_stop = [x - t0 for x in t_stop]
        xlim = [0, tspan]
        trial_numbers = [x - trial_numbers[0] + 1 for x in trial_numbers]
    else:
        t0 = 0
        xlim = [t_start, t_start+tspan]

    fname = 'population_activity_passive.pdf'
    plot_population_act(C, t_passive[plt_range]-t0, hd_theo[plt_range], decoded_direction_passive[plt_range], peak_firing_dir, fname, vmax=vmax, xlim=xlim, t_start=t_start, t_stop=t_stop, trial_numbers=trial_numbers, show_legend=False, fig_path=fig_path)

    t = data['time']
    passive_start = np.where(t[passive] > data['stimulus_start'][0])[0][0]
    passive_end = np.where(t[passive] > data['stimulus_stop'][-1])[0][0]
    plot_directions_and_drift(t, hd, decoded_direction, drift, passive, passive_start, passive_end, fig_path=fig_path)


def main():
    plot_platform_velocity(fig_path=os.path.join(BASE_PATH, 'examples'))

    session1 = 'TD0200_20231023_combined'
    combined_example(session1)

    session2 = 'TD0194_20230925_active_3landmarks'
    frmd7_active_example(session2)


if __name__ == "__main__":
    main()
