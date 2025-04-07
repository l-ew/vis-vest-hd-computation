from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy import ndimage
from vis_vest_hd_computation.utils import comp_prop_factor, convert_pvalue_to_asterisks, convert_pvalue_str, normalize_restricted, get_head_trajectory, align_restricted, align_rotations, avg_data, avg_data_per_animal, normalize_rotations
import matplotlib.markers as mmarkers
from vis_vest_hd_computation.imports import *
import itertools
import cmocean


def plot_restricted(df, var=DECODED_ROTATION, large_fig=False, display_neg=True, style='stim_type', folder=None, fig_path='.'):

    if folder is None:
        folder = var

    fig_path = os.path.join(fig_path, folder)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    fname = 'restricted_stimset.pdf'
    _plot_restricted(df, fname, fig_path, var=var, large_fig=large_fig, align_rotations=False, include_stim=True, display_neg=display_neg, normalized=False, style=style)

    fname = 'restricted_stimset_normalized_aligned.pdf'
    _plot_restricted(df, fname, fig_path, var=var, large_fig=large_fig, align_rotations=True, include_stim=False, display_neg=display_neg, normalized=True, style=style)


def _plot_restricted(df, fname, fig_path, large_fig=False, align_rotations=False, T=3, include_stim=True, display_neg=True, normalized=False, var=DECODED_ROTATION, style='stim_type', aggregate_by='animal'):

    if align_rotations:
        df = align_restricted(df, inplace=False)

    df = rename_cols(df.copy())
    palette = restricted_color_palette()

    if style == 'stim_type':
        markers = restricted_markers(align_rotations=align_rotations)
        dashes = False
        df = df.groupby([aggregate_by, 'vel', 'stim_type'])[var].agg('mean').reset_index()
    else:
        markers = '.'
        dashes = True  # {0: '', 1: (2, 2)}
        df = df.groupby([aggregate_by, 'vel', 'stim_type', style])[var].agg('mean').reset_index()

    if include_stim:
        df_indicated = pd.DataFrame()
        df_indicated['vel'] = np.unique(df['vel'])
        df_indicated[var] = np.unique(df['vel']) * T / 90
        df_indicated['stim_type'] = 'Stimulus'
        df = pd.concat([df, df_indicated], ignore_index=True)

    if normalized:
        normalize_restricted(df)
    else:
        df[var] *= 180 / np.pi

    if large_fig:
        fig = plt.figure(figsize=(5.0, 2.5))
    else:
        fig = plt.figure(figsize=(4.5, 2.25))
    ax1 = fig.add_axes([0.2, 0.225, 0.35, 0.7])

    ax = sns.lineplot(data=df, x="vel", y=var, hue='stim_type', estimator='mean', clip_on=False, legend='full', err_kws={'alpha': SHADING_ALPHA},
                      errorbar='se', ax=ax1, palette=palette, style=style, markers=markers, dashes=dashes)

    vels = np.unique(df['vel'])

    if align_rotations:
        ax.set_xlabel('Peak speed [deg/s]')
    else:
        ax.set_xlabel('Peak velocity [deg/s]')

    if normalized:

        if var == MODEL_ROTATION:
            ax.set_ylabel('Decoded HD shift\n(model) [%]')
        else:

            if var == MODEL_NEG_EYE_ROTATION:
                ax.set_ylabel('{} [%]'.format(var.replace("_", " ")), fontsize=SMALL_SIZE)
            else:
                ax.set_ylabel('{} [%]'.format(var.replace("_", " ")))

        if display_neg:
            ax.set_ylim(-35, 85)
            ax.set_yticks([-25, 0, 25, 50, 75])
            ax.axhline(0, color='k', linestyle='--', linewidth=1, zorder=-1)
        else:
            ax.set_ylim(-5,85)
            ax.set_yticks([0, 25, 50, 75])

        ax.set_xticks(vels)
        ax.set_xlim(35, 190)
    else:
        ax.hlines(0, xmin=-180, xmax=180, colors='k', linestyle='--', linewidth=1, zorder=-1)
        ax.set_ylabel('{} [deg]'.format(var).replace("_", " "))

        if not align_rotations:
            ax.set_xticks(vels[::2])
        else:
            ax.set_xticks(vels)
        ax.set_xlim(vels[0]-10, vels[-1]+10)
        ylim = np.max(np.abs(ax.get_ylim()))
        ax.set_ylim(-ylim, ylim)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = order_labels(df, labels)

    for ha in handles:
        ha.set_markeredgecolor('w')
        ha.set_markeredgewidth(0.75)

    if align_rotations:
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
        fig.savefig(os.path.join(fig_path, fname), dpi=300)
        plt.close(fig)
    else:
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1) 
        fig.savefig(os.path.join(fig_path, fname), dpi=300)
        plt.close(fig)


def scatter_decoded_rotation_vs_negative_eye_rotation_restricted(df, T=3, fname='scatterplot_decoded_rotation_vs_negative_eye_rotation.pdf', fig_path='.'):

    df = align_restricted(df.copy())
    df = rename_cols(df)
    df = df[np.logical_not(df['vel']==0)]
    normalize_restricted(df)

    palette = restricted_color_palette()
    markers = restricted_markers(scaling=0.85)

    df = df.groupby(['animal', 'vel', 'stim_type'])[[DECODED_ROTATION, NEG_EYE_ROTATION]].agg('mean').reset_index()
    df = df.sort_values(by=['animal', 'vel', 'stim_type'])

    fig = plt.figure(figsize=(5.5, 2.0))
    ax1 = fig.add_axes([0.1, 0.2, 0.3, 0.75])

    xmin = -10
    ax = sns.scatterplot(data=df, x=DECODED_ROTATION, y=NEG_EYE_ROTATION, hue='stim_type', ax=ax1, legend=False,
                         linewidth=0.25, alpha=1.0, style='stim_type', markers=markers, palette=palette, clip_on=False)

    ax.set_xticks([0,25,50,75])
    ax.set_yticks([0,25,50,75])
    ax.set_yticklabels([])
    ax.set_xlabel('{} [%]'.format(DECODED_ROTATION.replace("_", " ")))
    ax.set_ylabel('')
    ax.set_xlim(xmin,85)
    ax.set_ylim(-5,85)
    ax.set_aspect('equal', adjustable='datalim')

    x = df[DECODED_ROTATION].to_numpy().squeeze()
    y = df[NEG_EYE_ROTATION].to_numpy().squeeze()
    ax.plot([xmin, 85], [xmin, 85], 'k--',  zorder=0, clip_on=True)
    rho = np.corrcoef(x,y)[0,1]
    ax.text(75, 10, r'$\rho = ${:.2f}'.format(rho), fontsize=SMALL_SIZE)

    fig_path = os.path.join(fig_path, 'decoded_vs_negative_eye_rotation')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    fig.savefig(os.path.join(fig_path, fname), dpi=300)
    plt.close(fig)


def plot_all_combinations(df, var='rotation', large_fig=False, fig_path='.'):

    fig_path = os.path.join(fig_path, var)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    normalize = True
    align = True
    fname = 'rotation_per_motor_vel_normalized={}_aligned={}.pdf'.format(normalize, align)
    _plot_all_combinations(df, var=var, fname=fname, figpath=fig_path, large_fig=large_fig, normalize=normalize, align=align, is_aligned=False)

    normalize = False
    align = False
    fname = 'rotation_per_motor_vel_normalized={}_aligned={}.pdf'.format(normalize, align)
    _plot_all_combinations(df, var=var, fname=fname, figpath=fig_path, large_fig=False, normalize=normalize, align=align, is_aligned=False)

    normalize = True
    align = False
    fname = 'rotation_per_motor_vel_normalized={}_aligned={}.pdf'.format(normalize, align)
    _plot_all_combinations(df, var=var, fname=fname, figpath=fig_path, large_fig=large_fig, normalize=normalize, align=align, is_aligned=False)


def _plot_all_combinations(df, fname='all_combinations.pdf', var=DECODED_ROTATION, large_fig=False, figpath='trial_figs', T=3, aggregate_by='animal', align=True, is_aligned=True, normalize=False):

    df2 = df.copy()
    fixed = 'motor_velocity'

    if not is_aligned and align:
        align_rotations(df2, fixed)

    df2 = avg_data_per_animal(df2, var=var)

    if normalize:
        df2 = normalize_rotations(df2, fixed, invert=False, drop_invalid=True)

    if large_fig:
        fig = plt.figure(figsize=(3.75, 2.5))
    else:
        fig = plt.figure(figsize=(3.5, 2.25))
    ax1 = fig.add_axes([0.15, 0.225, 0.6, 0.7])

    palette = all_combinations_color_palette(align=align)

    errorbar = 'se'
    df_mean = df2.groupby(['motor_velocity', 'grating_velocity'])[var].agg('mean').reset_index()

    if not (align or is_aligned):
        vels = 45 * np.arange(-4,5)
        dashes = [(5,3)] * 4 + [(1,1)] + [(1,0)] * 4
        dashes = dict(zip(vels, dashes))

        hue_order = vels[::-1]

        ax = sns.lineplot(data=df2, x='grating_velocity', y=var, style=fixed, dashes=dashes, hue=fixed, hue_order=hue_order, estimator='mean', palette=palette, errorbar=errorbar, ax=ax1, markers=False, legend='full', err_kws={'alpha': SHADING_ALPHA}, clip_on=False)
    else:

        hue_order = 45 * np.arange(1,5)[::-1]

        ax = sns.lineplot(data=df2, x='grating_velocity', y=var, hue=fixed, hue_order=hue_order, estimator='mean', palette=palette, errorbar=errorbar, ax=ax1, markers=False, legend='full', err_kws={'alpha': SHADING_ALPHA}, clip_on=False)

    plt.scatter(df_mean['grating_velocity'], df_mean[var], s=18, linewidth=0.5, color=NEUTRAL_COLOR, edgecolors=NEUTRAL_EDGE_COLOR, zorder=3, clip_on=False)

    natural_mask = df_mean['grating_velocity'] == 0
    plt.scatter(df_mean['grating_velocity'][natural_mask], df_mean[var][natural_mask], s=18, linewidth=0.5, color=NATURAL_COLOR, edgecolors=NEUTRAL_EDGE_COLOR, zorder=4, clip_on=False)

    sync_mask = df_mean['grating_velocity'] == df_mean['motor_velocity']
    plt.scatter(df_mean['grating_velocity'][sync_mask], df_mean[var][sync_mask], s=18, linewidth=0.5, color=SYNC_COLOR, edgecolors=NEUTRAL_EDGE_COLOR, zorder=4, clip_on=False)

    ax.set_xlabel('Peak {} velocity [deg/s]'.format(SCENE_STR.replace('_', ' ')))
    ax.set_xticks(np.unique(df2['grating_velocity']))
    ax.axhline(0, color='k', linestyle='--', linewidth=1, zorder=-1)
    ax.axvline(0, color='k', linestyle='--', linewidth=1, zorder=-1)

    if normalize:
        if var == MODEL_NEG_EYE_ROTATION:
            ax.set_ylabel('{} [%]'.format(var.replace("_", " ")), fontsize=SMALL_SIZE)
        else:
            ax.set_ylabel('{} [%]'.format(var).replace("_", " "))

        ax.set_ylim(-35, 85)
        ax.set_yticks([-25, 0, 25, 50, 75])
    else:
        ax.set_yticks(np.pi / 180 * 100 * np.arange(-3,4))
        ax.set_ylim(-300 * np.pi / 180, 300 * np.pi / 180)
        ax.set_yticklabels(100 * np.arange(-3,4))
        ax.set_ylabel('{} [deg]'.format(var).replace("_", " "))

    if not is_aligned and not align:
        legend_fontsize = SMALL_SIZE
        bbox_to_anchor = (0.925, 1.15)
        loc = 'upper left'
    else:
        legend_fontsize = MEDIUM_SIZE
        bbox_to_anchor = (0.95, 0.5)
        loc = 'center left'

    if (align or is_aligned):
        vel_str = 'speed'
    else:
        vel_str = 'velocity'

    leg = ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, fontsize=legend_fontsize, ncol=1, title='Peak {} \n {} [deg/s]'.format(PLATFORM_STR.lower(), vel_str), frameon=False)
    leg.get_title().set_multialignment('center')

    a = ax1.get_xticks().tolist()
    a[0] = '-180 '
    ax1.set_xticklabels(a)

    fig.savefig(os.path.join(figpath, fname), dpi=300)
    plt.close(fig)


def scatterplot_all_combinations(df, xcol=DECODED_ROTATION, ycol=NEG_EYE_ROTATION, T=T_TRANSIENT_STIM, is_aligned=False, is_avg=False, include_visual_only=True, fname='scatterplot_decoded_rotation_vs_negative_eye_rotation_all_combinations.pdf', fig_path='.'):

    df2 = df.copy()
    fixed = 'motor_velocity'

    df_visual = df2[df2[fixed] == 0].copy()

    if not is_aligned:
        align_rotations(df2, fixed)

    if not is_avg:
        df2 = avg_data_per_animal(df2, var=[xcol, ycol])

    if include_visual_only:
        align_rotations(df_visual, 'grating_velocity')
        df_visual = avg_data_per_animal(df_visual, var=[xcol, ycol])

    df2 = normalize_rotations(df2, fixed, invert=False, drop_invalid=True)
    df_visual = normalize_rotations(df_visual, 'grating_velocity', invert=True, drop_invalid=True)

    fig = plt.figure(figsize=(2.5, 2.25))
    ax = fig.add_axes([0.2, 0.225, 0.7, 0.7])

    if ycol == MODEL_ROTATION:
        s = 15
    else:
        s = 10
    linewidth = 0.25

    ax = sns.scatterplot(data=df2, x=xcol, y=ycol, ax=ax, legend=False, linewidth=linewidth, alpha=1.0, color=NEUTRAL_COLOR, edgecolor=NEUTRAL_EDGE_COLOR, marker='o', s=s, clip_on=False)

    natural_mask = df2['grating_velocity'] == 0
    ax.scatter(df2[xcol][natural_mask], df2[ycol][natural_mask], s=s, linewidth=linewidth, color=NATURAL_COLOR, edgecolors=NEUTRAL_EDGE_COLOR, zorder=4, clip_on=False)

    sync_mask = df2['grating_velocity'] == df2['motor_velocity']
    ax.scatter(df2[xcol][sync_mask], df2[ycol][sync_mask], s=s, linewidth=linewidth, color=SYNC_COLOR, edgecolors=NEUTRAL_EDGE_COLOR, zorder=4, clip_on=False)

    if include_visual_only:
        ax.scatter(df_visual[xcol], df_visual[ycol], s=s, linewidth=linewidth, color=SCENE_COLOR, edgecolors=NEUTRAL_EDGE_COLOR, zorder=2, clip_on=False)

    if ycol == MODEL_ROTATION:
        ax.set_xlabel('{} (data) [%]'.format(xcol.replace('_', ' ')))
    else:
        ax.set_xlabel(xcol.replace('_', ' ') + ' [%]')
    ax.set_ylabel('')

    df_tot = pd.concat([df2, df_visual], ignore_index=True)

    xmin = -60
    xmax = 100

    x = df_tot[xcol].to_numpy().squeeze()
    y = df_tot[ycol].to_numpy().squeeze()

    ax.plot([xmin, xmax], [xmin, xmax], 'k--', lw=1, zorder=5, clip_on=True)
    rho = np.corrcoef(x,y)[0,1]
    ax.text(50, 10, r'$\rho = ${:.2f}'.format(rho), fontsize=SMALL_SIZE)

    ax.axhline(0, color='k', linestyle='--', linewidth=1, zorder=5)
    ax.axvline(0, color='k', linestyle='--', linewidth=1, zorder=5)

    ax.set_ylim(-35, 85)
    ax.set_yticks([-25, 0, 25, 50, 75])

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    y_range = y_limits[1] - y_limits[0]
    x_center = sum(x_limits) / 2
    aspect_ratio = fig.get_size_inches()[1] / fig.get_size_inches()[0]
    new_x_range = y_range / aspect_ratio  # Adjust x range to match y scaling
    ax.set_xlim(x_center - new_x_range / 2, x_center + new_x_range / 2)

    ax.set_xticks([-50, -25, 0, 25, 50, 75, 100])

    fig.savefig(os.path.join(fig_path, fname), dpi=300)
    plt.close(fig)


def plot_multiple_rotation_time_series(df, df_eye, df_model, df_model_eye, T=3, title='', txt=False, fname='rotation_over_time.pdf', figpath='.', offset=1):

    if txt:
        fig = plt.figure(figsize=(1.25,0.85))
        ax = fig.add_axes([0.3,.325,.65,.65])
    else:
        fig = plt.figure(figsize=(0.95,0.7))
        ax = fig.add_axes([0.25,.05,.75,.6])

    lw = 0.5

    rotation_deg = '{}_deg'.format(DECODED_ROTATION)
    df[rotation_deg] = df[DECODED_ROTATION] * 180 / np.pi
    ax1 = sns.lineplot(data=df, x='t', y=rotation_deg, estimator='mean', legend=False, errorbar='se', markers=False, linewidth=lw, ax=ax, color=DECODING_COLOR)

    if df_eye is not None:
        rotation_deg = '{}_deg'.format(NEG_EYE_ROTATION)
        df_eye[rotation_deg] = df_eye[NEG_EYE_ROTATION] * 180 / np.pi
        ax2 = sns.lineplot(data=df_eye, x='t', y=rotation_deg, estimator='mean', legend=False, errorbar='se', linewidth=lw, markers=False, ax=ax, color='k')

    if df_model_eye is not None:
        rotation_deg = '{}_deg'.format(MODEL_NEG_EYE_ROTATION)
        df_model_eye[rotation_deg] = df_model_eye[MODEL_NEG_EYE_ROTATION] * 180 / np.pi
        ax4 = sns.lineplot(data=df_model_eye, x='t', y=rotation_deg, estimator='mean', legend=False, errorbar=None, linewidth=lw, markers=False, ax=ax, color='k', linestyle='dashed')

    if df_model is not None:
        rotation_deg = '{}_deg'.format(MODEL_ROTATION)
        df_model[rotation_deg] = df_model[MODEL_ROTATION] * 180 / np.pi
        ax3 = sns.lineplot(data=df_model, x='t', y=rotation_deg, estimator='mean', legend=False, errorbar=None, linewidth=lw, markers=False, ax=ax, color=DECODING_COLOR, linestyle='dashed')

        if '{}_adj'.format(MODEL_ROTATION) in df_model.columns:
            rotation_deg = '{}_adj_deg'.format(MODEL_ROTATION)
            df_model[rotation_deg] = df_model['{}_adj'.format(MODEL_ROTATION)] * 180 / np.pi
            sns.lineplot(data=df_model, x='t', y=rotation_deg, estimator='mean', legend=False, errorbar=None, linewidth=lw, markers=False, ax=ax, color='k', linestyle='dashed')

    ax.axvline(0, color='k', linewidth=lw, linestyle='--', zorder=-1)
    ax.axvline(T, color='k', linewidth=lw, linestyle='--', zorder=-1)

    ax.set_xticks(np.arange(T+1))
    if not txt:
        ax.set_title(title, fontsize=5, pad=3)
    if txt:
        ax.set_xlabel('t [s]', fontsize=6)
        ax.set_ylabel('Shift [deg]', fontsize=6)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
    ax.set_xlim(-offset, T+offset)
    ax.tick_params(axis='both', labelsize=5)
    ax.tick_params('both', length=1.5, which='major')

    fig.savefig(os.path.join(figpath, fname), dpi=300)
    plt.close(fig)


def plot_eye_velocity(df, title='', vel_type=EYE_VELOCITY, fname='eye_velocity.pdf', figpath='.', T=3, offset=0, aggregate_by='animal'):

    df[vel_type] *= 180 / np.pi

    fig, ax = plt.subplots(1,1, figsize=(2.5,1.75), tight_layout=True)
    palette = sns.color_palette("coolwarm", n_colors=len(df['grating_velocity'].unique()))

    errorbar = None  # 'se'
    ax1 = sns.lineplot(data=df, x='t', y=vel_type, hue='grating_velocity', estimator='mean', palette=palette, legend=False, errorbar=errorbar, markers=False, ax=ax)

    max_speed = np.nanmax(np.abs(df[vel_type]))
    nticks = np.ceil(max_speed / 45)
    max_speed_tick = 45 * nticks
    ax.set_title(title)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('{} [deg/s]'.format(EYE_VELOCITY.replace('_', ' ')))
    ax.grid(True, which='both')
    ax.set_yticks(45 * np.arange(-nticks, nticks+1))
    ax.set_xlim(-offset, T+offset)
    ax.set_ylim(-max_speed_tick, max_speed_tick)
    ax.set_xticks(np.arange(T+1))

    fig.savefig(os.path.join(figpath, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def scene_vel_legend(fname='scene_vel_legend.pdf', figpath='.'):
    vels = 45 * np.arange(-4,5)
    colors = sns.color_palette("coolwarm", n_colors=len(vels))
    palette = dict(zip(vels, colors))
    fig, ax = plt.subplots(1,1, figsize=(1.5,2.0), tight_layout=False)
    custom_handles = [plt.Line2D([0], [0], color=color) for color in palette.values()]
    plt.axis('off')
    legend = ax.legend(custom_handles, palette.keys(), title='Peak {}\nvelocity [deg/s]'.format(SCENE_STR.lower().replace('_', ' ')))
    #plt.setp(legend.get_title(), loc='center')
    fig.savefig(os.path.join(figpath, fname), dpi=300)
    plt.close(fig)


def plot_rotation_time_series_all_stim(df_decoded, df_model, df_eye_pos, df_model_eye_pos, dark=False, txt=False, fig_path='.'):

    motor_vels = df_decoded['motor_velocity'].unique()
    grating_vels = df_decoded['grating_velocity'].unique()

    for vm in motor_vels:
        for vg in grating_vels:

            fname = 'motor={}_grating={}.pdf'.format(vm, vg)

            if dark:
                title = HEAD_VAR + ' = {} deg/s'.format(vm)
            else:
                title = HEAD_VAR + ' = {} deg/s\n'.format(vm) + SCENE_VAR + ' = {} deg/s'.format(vg)

            mask = np.logical_and(df_decoded['motor_velocity']==vm, df_decoded['grating_velocity']==vg)
            df_stim = df_decoded[mask].copy()

            mask = np.logical_and(df_model['motor_velocity']==vm, df_model['grating_velocity']==vg)
            df_model_stim = df_model[mask].copy()

            if df_model_eye_pos is not None:
                mask = np.logical_and(df_model_eye_pos['motor_velocity']==vm, df_model_eye_pos['grating_velocity']==vg)
                df_model_eye_stim = df_model_eye_pos[mask].copy()
            else:
                df_model_eye_stim = None

            if df_eye_pos is not None:
                mask = np.logical_and(df_eye_pos['motor_velocity']==vm, df_eye_pos['grating_velocity']==vg)
                df_eye_stim = df_eye_pos[mask].copy()
            else:
                df_eye_stim = None

            plot_multiple_rotation_time_series(df_stim, df_eye_stim, df_model_stim, df_model_eye_stim, title=title, fname=fname, txt=txt, figpath=fig_path)


def scatter_decoded_vs_eye_rotation(df, alpha, fig_path):
    fig, ax = plt.subplots(1,1, figsize=(2,2))
    ax.scatter(180 / np.pi * df[DECODED_ROTATION], 180 / np.pi * df[NEG_EYE_ROTATION], s=1)
    lim = 360
    ax.plot([-lim, lim], [-lim, lim], 'k--')
    ax.plot([-lim, lim], [-alpha * lim, alpha * lim], 'r--')
    ax.set_xlabel('{} [deg]'.format(DECODED_ROTATION.replace('_', ' ')))
    ax.set_ylabel('{} [deg]'.format(NEG_EYE_ROTATION.replace('_', ' ')))
    ax.set_xticks([-lim, 0, lim])
    ax.set_yticks([-lim, 0, lim])
    ax.axis('equal')
    fig.savefig(os.path.join(fig_path, 'eye_vs_decoded_rotation.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_statistic(df, statistic='statistic', unit='', p_value=None, fname='statistic.pdf', sizes=9, width=1.5, fig_path='.', category1='', category2='', category_type='stage', aggregate_by='animal', category_fontsize=SMALL_SIZE, ylim=100):

    fig, ax1 = plt.subplots(1,1, figsize=(width,width/1.5*1.75))
    plt.subplots_adjust(0.35, 0.2, 0.8, 0.9)

    animals = df[aggregate_by].unique()
    n_animals = len(animals)

    palette = dict(zip(animals, n_animals * ['darkgrey']))
    ax = sns.lineplot(data=df, x=category_type, y=statistic, hue=aggregate_by, palette=palette, zorder=0)

    palette = dict(zip(animals, n_animals * ['k']))
    ax = sns.scatterplot(data=df, x=category_type, y=statistic, hue=aggregate_by, marker='.', s=100, palette=palette, zorder=1)

    ax = sns.lineplot(data=df, x=category_type, y=statistic, estimator='mean', sizes=sizes, errorbar=None, c='red')
    ax.legend([], [], frameon=False)

    ax.set_xticks([0,1])
    if not category_type == 'rotation_dir':
        ax.set_xticklabels([category1, category2], fontsize=category_fontsize)
    ax.set_xlabel('')

    ax.set_xlim(-0.15, 1.15)

    ax.set_ylabel('{} [{}]'.format(statistic.replace('_', ' '), unit))
    if statistic == 'decoding_error':
        ylim = 100
        ax.set_ylabel('Decoding error [deg]')
        ax.set_yticks([0,50,100,150])
    elif statistic == 'vec_length':
        ylim = 1
        ax.set_ylabel('Vector length')
        ax.set_yticks([0,0.25,0.5,0.75])
    elif statistic == 'delta_rotation':
        ax.set_ylabel(r'$\Delta$ [deg]')
        ax.set_yticks([-45,0,45])
        ax.set_ylim(-70,70)
    elif statistic == 'internal_gain' or statistic == 'eye_gain':
        ax.set_yticks([0,25,50,75, 100])
        ax.set_ylim(0, ylim)

    if p_value is not None:
        pval = convert_pvalue_to_asterisks(p_value)
        plt.text(x=0.5, y=1.05*ax.get_ylim()[1], s=pval, horizontalalignment='center', verticalalignment='center', color='k', size=9)

    fig.savefig(os.path.join(fig_path, fname), dpi=300)
    plt.close(fig)


def plot_eye_velocity_data_vs_model(df, df_model, title='', fname='eye_velocity.pdf', figpath='.', T=3, offset=0, aggregate_by='animal'):

    df['eye_velocity_deg'] = df[EYE_VELOCITY] * 180 / np.pi
    df_model['eye_velocity_deg'] = df_model[EYE_VELOCITY] * 180 / np.pi

    fig, ax = plt.subplots(1,1, figsize=(2.5,1.75), tight_layout=True)

    ax1 = sns.lineplot(data=df, x='t', y='eye_velocity_deg', estimator='mean', errorbar='sd', markers=False, ax=ax, color='k')

    ax2 = sns.lineplot(data=df_model, x='t', y='eye_velocity_deg', estimator='mean', errorbar=None, markers=False, ax=ax, color=EYE_COLOR, linestyle='dashed')

    max_speed = np.nanmax(np.abs(df['eye_velocity_deg']))
    nticks = np.ceil(max_speed / 45)
    max_speed_tick = 45 * nticks

    ax.set_title(title)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('velocity [deg/s]')
    ax.grid(True, which='both')
    ax.set_yticks(45 * np.arange(-nticks, nticks+1))
    ax.set_xlim(-offset, T+offset)
    ax.set_ylim(-max_speed_tick, max_speed_tick)
    ax.set_xticks(np.arange(T+1))

    fig.savefig(os.path.join(figpath, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_unaligned_traces(t, internal_dirs, offset=1, title='', filename='decoded_direction.pdf', figpath='trial_figs', T=3):
    fig, ax = plt.subplots(1,1, figsize=(3,2))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(t)]

    for tt, internal_dir, color in zip(t, internal_dirs, colors):
        plot_offset_traces(ax, tt, internal_dir, c=color, alpha=1, zorder=2)

    ax.vlines(0, ymin=-np.pi, ymax=np.pi, color='k', linestyle='--')
    ax.vlines(T, ymin=-np.pi, ymax=np.pi, color='k', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('{} [rad]'.format(DECODED_ROTATION.replace("_", " ")))

    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.set_xlim(-offset, T+offset)
    ax.set_ylim(-np.pi, np.pi)

    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    fig.savefig(os.path.join(figpath, 'unaligned', filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_aligned_traces(t, internal_dirs, ref_dirs, title='', filename='decoded_direction_aligned.pdf', figpath='trial_figs', offset=1, T=3):
    fig, ax = plt.subplots(1,1, figsize=(3,2))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(t)]

    for tt, internal_dir, ref_dir, color in zip(t, internal_dirs, ref_dirs, colors):
        y = align_traces(internal_dir, ref_dir)
        plot_offset_traces(ax, tt, y, c=color, alpha=1, zorder=2)

    ax.vlines(0, ymin=-np.pi, ymax=np.pi, color='k', linestyle='--')
    ax.vlines(T, ymin=-np.pi, ymax=np.pi, color='k', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('{} [rad]'.format(DECODED_ROTATION.replace("_", " ")))

    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.set_xlim(-offset, T+offset)
    ax.set_ylim(-np.pi, np.pi)

    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    fig.savefig(os.path.join(figpath, 'aligned', filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_aligned_traces_with_eyepos(t, internal_dirs, ref_dirs, t_eye, eye_positions, reliable, saccade_start, saccade_stop, title='', filename='internal_dir_aligned.pdf', figpath='trial_figs', offset=1, T=3):

    for trial, (t_trial, internal_dir, ref_dir, t_eye_trial, eye_pos, ti_saccade, tf_saccade, r) in enumerate(zip(t, internal_dirs, ref_dirs, t_eye, eye_positions, saccade_start, saccade_stop, reliable)):

        fig, axs = plt.subplots(2,1, figsize=(5,4.5))

        y = align_traces(internal_dir, ref_dir)
        plot_offset_traces(axs[1], t_trial, y, c='k', alpha=1)

        # ki = np.where(t_eye_trial > 0)[0][0]
        # kf = np.where(t_eye_trial > T)[0][0]

        if r:

            axs[0].plot(t_eye_trial, eye_pos * 180 / np.pi, c='k', zorder=1)

            for start, stop in zip(ti_saccade, tf_saccade):
                start_ind = np.argmin(np.abs(t_eye_trial - start))
                end_ind = np.argmin(np.abs(t_eye_trial - stop))
                axs[0].scatter([start], [eye_pos[start_ind] * 180 / np.pi], marker='>', c='k', edgecolors='k', s=15, linewidth=1, zorder=2)
                axs[0].scatter([stop], [eye_pos[end_ind] * 180 / np.pi], marker='s', c='k', edgecolors='k', s=10, linewidth=1, zorder=2)

        ylim_eye = 30
        axs[0].vlines(0, ymin=-ylim_eye, ymax=ylim_eye, color='k', linestyle='--')
        axs[0].vlines(T, ymin=-ylim_eye, ymax=ylim_eye, color='k', linestyle='--')
        axs[0].set_title(title)
        axs[0].set_ylabel('eye position [deg]')

        axs[0].set_yticks([-30, -15, 0, 15, 30])
        axs[0].set_xlim(-offset, T+offset)
        axs[0].set_ylim(-ylim_eye, ylim_eye);

        axs[1].vlines(0, ymin=-np.pi, ymax=np.pi, color='k', linestyle='--')
        axs[1].vlines(T, ymin=-np.pi, ymax=np.pi, color='k', linestyle='--')
        axs[1].set_xlabel('t [s]')
        axs[1].set_ylabel('internal direction [rad]')

        axs[1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axs[1].set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        axs[1].set_xlim(-offset, T+offset)
        axs[1].set_ylim(-np.pi, np.pi);

        fname, ext = os.path.splitext(filename)
        fig.savefig(os.path.join(figpath, 'aligned', '{}_trial={}.{}'.format(fname, trial, ext)), dpi=300, bbox_inches='tight')
        plt.close(fig)


def overlay_aligned_traces(data_ccw, data_cw, data_catch=None, title='', filename='decoded_direction_overlayed.pdf', figpath='trial_figs', offset=1, T=3, t_theo=None, x_theo=None):
    fig, ax = plt.subplots(1,1, figsize=(3,2), tight_layout=True)

    for t, internal_dir, before in zip(data_cw['t'], data_cw[DECODED_DIRECTION], data_cw['before']):
        y = align_traces(internal_dir, before)
        plot_offset_traces(ax, t, y, c='r', alpha=0.33, zorder=2)

    for t, internal_dir, before in zip(data_ccw['t'], data_ccw[DECODED_DIRECTION], data_ccw['before']):
        y = align_traces(internal_dir, before)
        plot_offset_traces(ax, t, y, c='b', alpha=0.33, zorder=2)

    if data_catch is not None:
        for t, internal_dir, before in zip(data_catch['t'], data_catch[DECODED_DIRECTION], data_catch['before']):
            y = align_traces(internal_dir, before)
            plot_offset_traces(ax, t, y, c='y', alpha=0.33, zorder=1)

    if t_theo is not None:
        for delta in [-2*np.pi, 0, 2*np.pi]:
            ax.plot(t_theo, x_theo+delta, 'k--', zorder=3)
            ax.plot(t_theo, -x_theo+delta, 'k--', zorder=3)

    ax.vlines(0, ymin=-np.pi, ymax=np.pi, color='k', linestyle='--', zorder=0)
    ax.vlines(T, ymin=-np.pi, ymax=np.pi, color='k', linestyle='--', zorder=0)
    ax.set_title(title)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('{} [rad]'.format(DECODED_DIRECTION.replace('_', ' ')))

    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.set_xlim(-offset, T+offset)
    ax.set_ylim(-np.pi, np.pi)

    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    fig.savefig(os.path.join(figpath, 'overlayed', filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_offset_traces(ax, t, y, c='k', alpha=0.33, zorder=1):
    for offset in [-2*np.pi, 0, 2*np.pi]:
        ax.plot(t, y+offset, c=c, alpha=alpha, zorder=zorder)


def align_traces(internal_dir, ref_dir):
    y = internal_dir.copy() - ref_dir
    return y


def direction_str(x):
    if x == 1:
        return 'cw'
    elif x == -1:
        return 'ccw'
    else:
        return ''


def rename_cols(df):
    if 'dark' in df.columns and df['dark'].any():
        df.loc[df['dark'], 'stim_type'] = df.loc[df['dark'], 'stim_type'].replace([PLATFORM_STR], DARK_ROT_LONG)
        mask = np.logical_not(df['dark'])
    else:
        mask = np.ones(df.shape[0], dtype=bool)
    df.loc[mask, 'stim_type'] = df.loc[mask, 'stim_type'].replace([PLATFORM_STR], NATURAL_ROT_LONG)
    df.loc[mask, 'stim_type'] = df.loc[mask, 'stim_type'].replace([SCENE_STR], SCENE_ROT_LONG)
    df.loc[mask, 'stim_type'] = df.loc[mask, 'stim_type'].replace([SYNC_STR], SYNC_ROT_LONG)
    return df


def order_labels(df, labels):
    ordered_labels = ['Stimulus', NATURAL_ROT_LONG, SYNC_ROT_LONG, SCENE_ROT_LONG, DARK_ROT_LONG]
    order = [labels.index(x) for x in ordered_labels if x in labels]
    return order


def animal_palette():
    animals = ['TD0150', 'TD0156', 'TD0161', 'TD0175', 'TD0180', 'TD0199', 'TD0200', 'TD0201', 'TD0203', 'TD0204']
    animal_colors = sns.color_palette("muted", len(animals))
    palette = dict(zip(animals, animal_colors))
    return palette


def restricted_color_palette():
    stim_types = [NATURAL_ROT_LONG, SYNC_ROT_LONG, SCENE_ROT_LONG, DARK_ROT_LONG, 'Stimulus']
    stim_colors = [NATURAL_COLOR, SYNC_COLOR, SCENE_COLOR, DARK_COLOR, STIM_COLOR]
    palette = dict(zip(stim_types, stim_colors))
    return palette


def restricted_markers(align_rotations=True, scaling=1.0):
    stim_types = [NATURAL_ROT_LONG, SCENE_ROT_LONG, SYNC_ROT_LONG, DARK_ROT_LONG, 'Stimulus']
    if align_rotations:
        markers = dict(zip(stim_types, [mmarkers.MarkerStyle('.').scaled(scaling),
                                        mmarkers.MarkerStyle('v').scaled(0.65*scaling),
                                        mmarkers.MarkerStyle('^').scaled(0.65*scaling),
                                        mmarkers.MarkerStyle('s').scaled(0.4*scaling),
                                        mmarkers.MarkerStyle('.').scaled(scaling)]))
    else:
        markers = dict(zip(stim_types, [mmarkers.MarkerStyle('.').scaled(0.8), mmarkers.MarkerStyle('v').scaled(0.5), mmarkers.MarkerStyle('^').scaled(0.5), mmarkers.MarkerStyle('s').scaled(0.35), mmarkers.MarkerStyle('.').scaled(0.8)]))
    return markers


def xy_axis(ax, vels, align_rotations, var='internal rotation'):
    ax.set_xlabel('peak velocity [deg/s]')
    ax.set_ylabel('{} [rad]'.format(var))
    ax.set_xticks(vels)
    ax.set_xlim(vels[0]-10, vels[-1]+10)
    if align_rotations:
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3 \pi}{2}$', r'$2\pi$'])
        ax.set_ylim(-np.pi/4, 2*np.pi+np.pi/4)
    else:
        ax.set_yticks([-2*np.pi,-np.pi,0, np.pi, 2*np.pi])
        ax.set_yticklabels([r'$-2\pi$', r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$'])
        ax.set_ylim(-2*np.pi-np.pi/4, 2*np.pi+np.pi/4)


def plot_restricted_model_frmd7(df, fig_path, T=3, display_neg=True, var=MODEL_ROTATION, style='stim_type', aggregate_by='animal'):

    df = align_restricted(df, inplace=False)
    df = rename_cols(df.copy())
    df = df.groupby([aggregate_by, 'vel', 'stim_type'])[var].agg('mean').reset_index()
    normalize_restricted(df)

    fig = plt.figure(figsize=(5.0, 2.5))
    ax = fig.add_axes([0.2, 0.225, 0.35, 0.7])
    dash_pattern = (5,5)
    markers = restricted_markers()

    if display_neg:
        ymin = -35
        ymax = 85
        yrange = 120
        yticks = np.array([-25, 0, 25, 50, 75])
    else:
        ymin = -5
        ymax = 85
        yrange = 90
        yticks = np.array([0, 25, 50, 75])

    xrange = 190 - 35
    scale = xrange / yrange

    df[var] *= scale

    mask = df['stim_type'] == SCENE_ROT_LONG
    ax.plot(df['vel'][mask], df[var][mask], color=SCENE_COLOR, lw=1.5)
    ax.scatter(df['vel'][mask], df[var][mask], color=SCENE_COLOR, marker=markers[SCENE_ROT_LONG], edgecolor='w', linewidth=0.75, zorder=2)

    vels = np.unique(df['vel'])
    ax.set_xlabel('Peak speed [deg/s]')
    ax.set_ylabel('{} [%]'.format(DECODED_ROTATION.replace("_", " ")))
    ax.set_ylim(ymin * scale, ymax * scale)
    ax.set_yticks(scale * yticks)
    ax.set_yticklabels(yticks)
    ax.set_xticks(vels)
    ax.set_xlim(35, 190)

    from matplotlib.patches import Wedge, Circle

    mask = df['stim_type'] == SYNC_ROT_LONG
    ax.plot(df['vel'][mask], df[var][mask], color=SYNC_COLOR, linestyle=(0, dash_pattern), lw=1.5)

    mask = df['stim_type'] == NATURAL_ROT_LONG
    ax.plot(df['vel'][mask], df[var][mask], color=NATURAL_COLOR, linestyle=(5, dash_pattern), lw=1.5)

    for v in [45, 90, 135, 180]:
        mask = np.logical_and(df['stim_type'] == NATURAL_ROT_LONG, df['vel'] == v)
        y = float(df[var][mask])
        r = 3.5
        wedge1 = Wedge(center=(v, y), r=r, theta1=45, theta2=225, facecolor=NATURAL_COLOR, zorder=2)
        wedge2 = Wedge(center=(v, y), r=r, theta1=225, theta2=45, facecolor=SYNC_COLOR, zorder=2)
        circle = Circle((v, y), r, facecolor='none', edgecolor='w', linewidth=0.75, zorder=3)
        ax.add_patch(wedge1)
        ax.add_patch(wedge2)
        ax.add_patch(circle)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = order_labels(df, labels)

    for ha in handles:
        ha.set_markeredgecolor('w')
        ha.set_markeredgewidth(0.75)

    fig_path = os.path.join(fig_path, var)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    fname = 'restricted_stimset_normalized_aligned.pdf'
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
    fig.savefig(os.path.join(fig_path, fname), dpi=300)
    plt.close(fig)


def close_gap(ax0, ax1):
    pos1 = ax0.get_position()
    pos2 = ax1.get_position()
    gap = pos1.y0 - pos2.y1
    ax1.set_position([pos2.x0, pos2.y0 + 0.25 * gap, pos2.width, pos2.height], which='both')


def plot_eyepos_and_rotation(data_stim, title='', filename='eye_pos_and_rotation.pdf', figpath='trial_figs', offset=1, T=3, fps=60, orientation=1):

    ticks_eyepos = [-15, 0, 15]
    ylim_eye_pos = 25
    if data_stim['motor_velocity'] == 0:
        ticks_shift = [0, 60]
        ylim_shift = [-15, 75]
    elif data_stim['motor_velocity'] == 1:
        ticks_shift = [0, 60]
        ylim_shift = [-15, 90]
    elif data_stim['motor_velocity'] == 2:
        ticks_shift = [0, 90, 180]
        ylim_shift = [-15, 180]
    else:
        ticks_shift = [0, 180, 360]
        ylim_shift = [-15, 380]

    if data_stim['motor_direction'] == 1 or data_stim['motor_direction'] == 0 and data_stim['grating_direction'] == -1:
        ticks_shift = [-x for x in ticks_shift[::-1]]
        ylim_shift = [-x for x in ylim_shift[::-1]]

    dt = 1 / fps

    if data_stim['motor_velocity'] == 0 and data_stim['grating_velocity'] != 0:
        color = SCENE_COLOR
    elif data_stim['motor_velocity'] == data_stim['grating_velocity'] and data_stim['motor_direction'] == data_stim['grating_direction'] and data_stim['grating_velocity'] != 0:
        color = SYNC_COLOR
    elif data_stim['motor_velocity'] != 0 and data_stim['grating_velocity'] == 0:
        color = NATURAL_COLOR
    else:
        color = 'k'

    for k, (t, hd_est, t_eye, eye_pos, ti_saccade, tf_saccade, r, coef, coef2) in enumerate(zip(data_stim['t'], data_stim[DECODED_DIRECTION], data_stim['t_eye'], data_stim['eye_position'], data_stim['ti_qp'], data_stim['tf_qp'], data_stim['eye_reliable'], data_stim['eye_velocity_coef'], data_stim['eye_velocity_coef_2nd_half'])):

        fig, axs = plt.subplots(2,1, figsize=(1.4, 1.5), sharex=True)
        fig.subplots_adjust(left=0.25)
        fig.subplots_adjust(bottom=0.25)

        mask = np.logical_and(t_eye >= -1, t_eye <= 4)
        lines = axs[0].plot(t_eye[mask], eye_pos[mask] * 180 / np.pi, c=color, zorder=1, clip_on=True)
        lines[0].set_clip_path(None)
        start_inds = [np.argmin(np.abs(t_eye - x)) for x in ti_saccade]
        stop_inds = [np.argmin(np.abs(t_eye - x)) for x in tf_saccade]
        saccade_mask = np.zeros(len(t_eye), dtype=bool)
        for start_ind, stop_ind in zip(start_inds, stop_inds):
            axs[0].scatter(t_eye[start_ind], [eye_pos[start_ind] * 180 / np.pi], marker='>', color=color, edgecolors='k', s=8, clip_on=True, linewidth=1, zorder=2)
            axs[0].scatter(t_eye[stop_ind], [eye_pos[stop_ind] * 180 / np.pi], marker='s', color=color, edgecolors='k', s=5, clip_on=True, linewidth=1, zorder=2)
            saccade_mask[start_ind:stop_ind+1] = True

        axs[0].get_xaxis().set_visible(False)
        axs[0].spines['bottom'].set_visible(False)
        axs[0].set_yticks(ticks_eyepos)
        axs[0].set_xlim(-offset, T+offset)
        axs[0].set_ylim(-ylim_eye_pos, ylim_eye_pos);

        close_gap(axs[0], axs[1])

        if r:
            if len(coef2) == 0:
                int_coef = [0] + [x / (k+1) for k, x in enumerate(coef)]
                p = Polynomial(int_coef)
                mask = np.logical_and(t_eye > 0, t_eye < T)
                eye_rotation = p(t_eye[mask])
            else:

                def pw_poly(x, coef, coef2, T):
                    int_coef = [0] + [x / (k+1) for k, x in enumerate(coef)]
                    p1 = Polynomial(int_coef)
                    int_coef2 = [0] + [x / (k+1) for k, x in enumerate(coef2)]
                    p2 = Polynomial(int_coef2)
                    eye_rotation = np.zeros_like(x)
                    mask1 = x < 0.5 * T
                    mask2 = x >= 0.5 * T
                    eye_rotation[mask1] = p1(x[mask1])
                    eye_rotation[mask2] = p1(0.5 * T) + p2(x[mask2]) - p2(0.5 * T)
                    return eye_rotation

                p = lambda x: pw_poly(x, coef, coef2, T)
                mask = np.logical_and(t_eye > 0, t_eye < T)
                eye_rotation = p(t_eye[mask])

            axs[1].plot(t_eye[mask], eye_rotation * 180 / np.pi, c='k', zorder=3, linestyle=':')

            unwrapped_eyepos = eye_pos.copy()
            unwrapped_eyepos[saccade_mask] = np.nan
            k0 = np.argmin(np.abs(t_eye))
            unwrapped_eyepos -= unwrapped_eyepos[k0]
            unwrapped_eyepos[:k0] = np.nan
            kf = np.argmin(np.abs(t_eye - T))
            unwrapped_eyepos[kf:] = np.nan

            for ti, tf, ki, kf in zip(ti_saccade, tf_saccade, start_inds, stop_inds):
                unwrapped_eyepos[kf + 1:] += unwrapped_eyepos[ki - 1] - unwrapped_eyepos[kf + 1]
                unwrapped_eyepos[kf + 1:] += p(tf + dt) - p(ti - dt)

            axs[1].plot(t_eye, unwrapped_eyepos * 180 / np.pi, c=color, zorder=2)

        axs[1].set_yticks(ticks_shift)
        axs[1].set_ylim(ylim_shift)
        axs[1].set_xlim(-offset, T+offset)
        axs[1].set_xticks([0,1,2,3])
        axs[1].set_xlabel('t [s]', fontsize=7)
        axs[1].axvline(0, ymin=0, ymax=2.2, color='k', linestyle='--', clip_on=False)
        axs[1].axvline(T, ymin=0, ymax=2.2, color='k', linestyle='--', clip_on=False)

        for spine in axs[1].spines.values():
            spine.set_zorder(0)

        fig.savefig(os.path.join(figpath, 'trial={}.pdf'.format(k)), dpi=300)
        plt.close(fig)


def plot_circ_trace(ax, t, x, c='tab:blue', label='', linewidth=1):
    mask = np.isfinite(x)
    x = x[mask]
    t = t[mask]

    indices = np.where(np.abs(np.diff(x)) > np.pi)[0] + 1
    for n, i in enumerate(indices):
        if n == 0:
            ax.plot(t[:i], x[:i], color=c, linestyle='-', label=label, linewidth=linewidth)
        else:
            ax.plot(t[indices[n - 1]:i], x[indices[n - 1]:i], color=c, linestyle='-', linewidth=linewidth)
    if len(indices) > 0:
        ax.plot(t[indices[-1]:], x[indices[-1]:], color=c, linestyle='-', linewidth=linewidth)
    else:
        ax.plot(t, x, color=c, linestyle='-', label=label, linewidth=linewidth)


def plot_palette(palette, title="Peak scene\nvelocity [deg/s]", figpath='.'):
    fig = plt.figure(figsize=(1.25,1.5))
    ax = fig.add_axes([0.325,.275,.65,.6])
    from matplotlib.lines import Line2D
    legend_lines = [Line2D([0], [0], color=palette[key], linewidth=2) for key in palette.keys()]
    legend = ax.legend(legend_lines, palette.keys(), title="Peak scene\nvelocity [deg/s]")
    title = legend.get_title()
    title.set_horizontalalignment('center')
    ax.axis("off")
    fig.savefig(os.path.join(figpath, 'palette.pdf'), dpi=300)
    plt.close(fig)


def adjust_ylabel_fontsize(fig, ax, fontsize=6):
    ylabel_box = ax.yaxis.get_label().get_window_extent()
    fig_width, fig_height = fig.get_size_inches() * fig.dpi

    if ylabel_box.height > 0.9 * fig_height:
        print(ylabel_box.height)
        ax.yaxis.label.set_fontsize(fontsize)


def plot_rotation_time_series(df, var=DECODED_ROTATION, color='k', cond='natural', style='motor_velocity', palette=False, dashes=VEL_DASHES, figpath='.'):

    figpath = os.path.join(figpath, cond)
    os.makedirs(figpath, exist_ok=True)

    if cond == 'dark' or cond == 'natural':
        ylim = [-25, 275]
        yticks = [0, 100, 200]
    elif cond == 'sync':
        ylim = [-25, 200]
        yticks = [0, 100, 200]
    elif cond == 'true':
        ylim = [-25, 360]
        yticks = [0, 100, 200, 300]
    else:
        ylim = None
        yticks = None

    fname = 'all_velocities.pdf'
    _plot_rotation_time_series(df, var=var, style=style, color=color, palette=palette, dashes=dashes, ylim=ylim, yticks=yticks, fname=fname, figpath=figpath)

    if palette:
        plot_palette(palette, title="Peak scene\nvelocity [deg/s]", figpath=figpath)

    ylim = None
    yticks = None
    for vmax in [0, 45, 90, 135, 180]:
        fname = 'vmax={}.pdf'.format(vmax)
        df2 = df[df[style] == vmax].copy()
        _plot_rotation_time_series(df2, var=var, style=style, color=color, palette=palette, ylim=ylim, yticks=yticks, fname=fname, dashes={vmax: ()}, figpath=figpath)


def _plot_rotation_time_series(df, var=DECODED_ROTATION, T=3, offset=1, color='k', style='motor_velocity', palette=False, ylim=None, yticks=None, dashes=VEL_DASHES, fname='rotation_over_time.pdf', figpath='.'):

    fig = plt.figure(figsize=(1.75,1.3))
    ax = fig.add_axes([0.375,.28,.6,.65])

    rotation_deg = '{}_deg'.format(var)
    df[rotation_deg] = df[var] * 180 / np.pi

    if not palette:
        ax1 = sns.lineplot(data=df, x='t', y=rotation_deg, style=style, color=color, dashes=dashes, estimator='mean', legend=False, errorbar='se', err_kws={'alpha': SHADING_ALPHA}, markers=False, ax=ax)
    else:
        ax1 = sns.lineplot(data=df, x='t', y=rotation_deg, style=style, hue=style, palette=palette, dashes=dashes, estimator='mean', legend=False, errorbar='se', err_kws={'alpha': SHADING_ALPHA}, markers=False, ax=ax)

    ax.set_xlabel('t [s]')

    if var == DECODED_ROTATION:
        ax.set_ylabel('Decoded HD [deg]')
    else:
        ax.set_ylabel('{} [deg]'.format(var.replace("_", " ")))

    ax.set_xlim(-offset, T + offset)
    adjust_ylabel_fontsize(fig, ax, fontsize=6)

    if yticks is not None:
        ax.set_yticks(yticks)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()

    ax.axvline(0, color='k', linestyle='--', zorder=-1)
    ax.axvline(T, color='k', linestyle='--', zorder=-1)
    ax.set_xticks(np.arange(T+1))

    fig.savefig(os.path.join(figpath, fname), dpi=300)
    plt.close(fig)


def plot_decoded_and_true_rotation_time_series(df_true, df_decoded, T=3, offset=1, color='k', true_rotation=None, decoded_rotation=None, sem=None, ylim=None, yticks=None, fname='direction_over_time.pdf', figpath='.'):

    fig = plt.figure(figsize=(1.6,1.25))
    ax = fig.add_axes([0.3,.3,.6,.6])

    df_true['rotation_deg'] = df_true[TRUE_ROTATION] * 180 / np.pi
    df_decoded['rotation_deg'] = df_decoded[DECODED_ROTATION] * 180 / np.pi

    sns.lineplot(data=df_true, x='t', y='rotation_deg', color=HEAD_COLOR, estimator='mean', legend=False, errorbar='se', markers=False, ax=ax)
    sns.lineplot(data=df_decoded, x='t', y='rotation_deg', color=color, estimator='mean', legend=False, errorbar='se', err_kws={'alpha': SHADING_ALPHA}, markers=False, ax=ax)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()
        ax.set_ylim(-15, ylim[1])

    y_offset = 0.025 * (ylim[-1] - ylim[0])
    ax.text(T, true_rotation, '100%', fontsize=MEDIUM_SIZE, color=HEAD_COLOR, horizontalalignment='left', verticalalignment='bottom')
    if decoded_rotation is not None:
        rel_shift = 100 * decoded_rotation / true_rotation
        ax.text(T, decoded_rotation + y_offset, r' {:.0f}%'.format(rel_shift), color=color, fontsize=MEDIUM_SIZE, horizontalalignment='left', verticalalignment='bottom')
        #rel_err = 100 * sem / true_rotation
        #ax.text(T - 1.0, decoded_rotation + y_offset, r'{:.0f}%$\pm${:.0f}%'.format(rel_shift, rel_err), color=color, fontsize=MEDIUM_SIZE, horizontalalignment='left', verticalalignment='bottom')

    ax.set_xlabel('t [s]')
    ax.set_ylabel('Direction [deg]')
    ax.set_xlim(-offset, T+offset)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.axhline(0, color='k', linestyle='-', lw=0.5, zorder=-1)
    ax.axvline(0, color='k', linestyle='--', zorder=-1)
    ax.axvline(T, color='k', linestyle='--', zorder=-1)
    ax.set_xticks(np.arange(T+1))

    fig.savefig(os.path.join(figpath, fname), dpi=300)
    plt.close(fig)


def all_combinations_color_palette(align=True):
    if align:
        vels = 45 * np.arange(0,5)
        colors = sns.color_palette("GnBu_d", n_colors=len(vels))
        colors = colors[1:]
        vels = vels[1:]

    else:
        vels = 45 * np.arange(0,5)
        colors = sns.color_palette("GnBu_d", n_colors=len(vels))
        vels = np.concatenate((-vels[-1:0:-1], vels))
        colors = np.concatenate((colors[-1:0:-1], colors))

    palette = dict(zip(vels, colors))
    return palette


def plot_circ_corr(lag, circ_corr, fps=12, fname='circ_corr_vs_delay.pdf', fig_path='.'):

    mean_circ_corr = np.nanmean(circ_corr, axis=0)
    std_circ_corr = np.nanstd(circ_corr, axis=0, ddof=1)
    k = np.argmax(circ_corr, axis=-1)

    mask = np.all(np.isnan(circ_corr), axis=-1)
    t_max = np.array([lag[x] for x in k])
    t_max[mask] = np.nan
    t_max = np.nanmean(t_max)

    plt.figure(figsize=(1.25,1.0))
    plt.plot(lag, mean_circ_corr)
    plt.fill_between(lag, mean_circ_corr-std_circ_corr, mean_circ_corr+std_circ_corr, alpha=0.5)
    plt.ylabel('Circular correlation')
    plt.xlabel(r'$\Delta t$ [s]')
    plt.xlim(lag[0], lag[-1])
    plt.xticks(np.arange(-2,3))
    plt.vlines(0, ymin=0, ymax=1, color='k', linestyle='--')
    plt.ylim(0,1)
    plt.text(t_max + 0.1, 0.95, r"$\langle t_{max} \rangle = $" + f"{t_max:.2f} s", fontsize=7);
    plt.savefig(os.path.join(fig_path, fname), bbox_inches='tight')
    plt.close()


def mean_std_scatter(y, label=None, lim=None, ticks=None, fname='x.pdf', fig_path='.', horizontal=False):
    x = np.random.randn(len(y))

    if horizontal:
        fig, ax = plt.subplots(1,1, figsize=(1.25, 0.5))
        x, y = y, x
    else:
        fig, ax = plt.subplots(1,1, figsize=(0.75,1.75))

    ax.scatter(x, y, marker='o', edgecolor='None', facecolor=(1,1,0.76,0.5), s=25, zorder=0, clip_on=False)
    ax.scatter(x, y, marker='o', edgecolor='k', facecolor='None', s=25, zorder=0, linewidth=0.5, clip_on=False)

    if horizontal:
        ax.errorbar(np.nanmean(x), 7.5, xerr=np.nanstd(x, ddof=1), c='tab:red', fmt='.', capsize=5, elinewidth=1.5, markersize=7.5, capthick=1.5, zorder=2, clip_on=False)
    else:
        ax.errorbar(7.5, np.nanmean(y), yerr=np.nanstd(y, ddof=1), c='tab:red', fmt='.', capsize=5, elinewidth=1.5, markersize=7.5, capthick=1.5, zorder=2, clip_on=False)

    if horizontal:
        ax.set_ylim(-5,10)
        ax.set_yticks([])
        if lim is not None:
            ax.set_xlim(lim)
        if ticks is not None:
            ax.set_xticks(ticks)
        if label is not None:
            ax.set_xlabel(label, fontsize=MEDIUM_SIZE)
    else:
        ax.set_xlim(-5,10)
        ax.set_xticks([])
        if lim is not None:
            ax.set_ylim(lim)
        if ticks is not None:
            ax.set_yticks(ticks)
        if label is not None:
            ax.set_ylabel(label, fontsize=MEDIUM_SIZE)

    ax.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
    fig.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def pfd_hist2d(pfd1, pfd2, xlabel='', ylabel='', fname='hist2d_pfd.pdf', fig_path='.'):
    hd_bins = 18  
    hd_bin_edges = np.linspace(0, 2*np.pi, hd_bins + 1)

    fig, ax = plt.subplots(1,1, figsize=(1.5,1.5))
    h = ax.hist2d(pfd1, pfd2, bins=hd_bin_edges, cmap='BuPu')

    ax.set_aspect(1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0,2*np.pi)
    ax.set_ylim(0,2*np.pi)
    ax.set_xticks([0,np.pi,2*np.pi])
    ax.set_yticks([0,np.pi,2*np.pi])
    ax.set_xticklabels([0,180,360])
    ax.set_yticklabels([0,180,360])

    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.8, 0.3, 0.03, 0.4])
    fig.colorbar(h[3], cax=cbar_ax, label = 'Frequency')
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    fig.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def scatter_vec_length(r1, r2, xlabel='', ylabel='', fname='scatter_vec_length.pdf', figpath='.'):

    fig, ax = plt.subplots(1,1, figsize=(1.5,1.5))

    rho = np.corrcoef(r1, r2)[0,1]
    ax.text(0.1, 0.85, r'$\rho$ = {:.2f}'.format(rho))

    ax.plot([0,1], [0,1], 'r--', zorder=2)
    ax.scatter(r1, r2, s=1, alpha=0.5, color='k')
    ax.axis('square')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])

    fig.savefig(os.path.join(figpath, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def scatter_peak_activity(x, y, xlabel='', ylabel='', fname='scatter_peak_act.pdf', figpath='.'):
    fig, ax = plt.subplots(1,1, figsize=(1.5,1.5))
    ax.plot([30,1000], [30,1000], 'r--', zorder=2)
    ax.scatter(x, y, s=1, alpha=0.5, color='k')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axis('square')

    fig.savefig(os.path.join(figpath, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def vec_length_hist(vec_length, significant_tuning, stage, frmd7=False, fname='vec_length_hist.pdf', figpath='.'):

    fig = plt.figure(figsize=(1.8, 1.25))
    ax = fig.add_axes([0.25, 0.325, 0.6, 0.5])

    significant_vec_length = [x[stage][y[stage]] for x, y in zip(vec_length, significant_tuning)]
    significant_vec_length = list(itertools.chain(*significant_vec_length))

    nonsignificant_vec_length_stage = [x[stage][~y[stage]] for x, y in zip(vec_length, significant_tuning)]
    nonsignificant_vec_length_stage = list(itertools.chain(*nonsignificant_vec_length_stage))

    if frmd7:
        bins = np.linspace(0,1,13)
    else:
        bins = np.linspace(0,1,26)
    ax.hist([significant_vec_length, nonsignificant_vec_length_stage], color=['#070d0d', 'darkgrey'], bins=bins,
             density=False, histtype='bar', stacked=True, label=['significant', 'non-significant'])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.35, 1), loc='lower left', fontsize=TINY_SIZE, mode='expand', ncol=2)
    ax.set_ylabel('# neurons')
    ax.set_xlabel('Vector length')

    ylim = ax.get_ylim()
    if ylim[1] > 40 and ylim[1] < 50:
        ax.set_ylim([0, 50])
        ax.set_yticks([0, 25, 50])
    elif ylim[1] < 40:
        nticks = np.ceil(ylim[1] / 10)
        ax.set_ylim([0, 10 * nticks])
        ax.set_yticks(10 * np.arange(nticks+1))

    ax.set_xlim([0,1])
    ax.set_xticks([0, 0.5, 1])

    fig.savefig(os.path.join(figpath, fname), dpi=300)
    plt.close(fig)


def delta_pfd_hist(x, fname='delta_pfd.pdf', figpath='.'):
    fig = plt.figure(figsize=(1.45, 1.25))
    ax = fig.add_axes([0.3, 0.325, 0.6, 0.5])

    hd_bins = 36
    hd_bin_edges = np.linspace(-np.pi, np.pi, hd_bins + 1)
    bins = hd_bin_edges + 0.5 * (hd_bin_edges[1] - hd_bin_edges[0])
    bins = np.hstack((2 * bins[0] - bins[1], bins))

    ax.hist(x, color='#070d0d', bins=bins, density=False, histtype='bar')
    ax.set_ylabel('# neurons')
    ax.set_xlabel(r'$\Delta$ PHD [deg]')

    ax.set_xlim([-np.pi, np.pi])
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([-180, -90, 0, 90, 180], fontsize=6)

    fig.savefig(os.path.join(figpath, fname), dpi=300)
    plt.close(fig)


def scatter_decoded_vs_head_rotation(df, fname='scatter_decoded_vs_head_rotation.pdf', figpath='.'):

    fig, ax = plt.subplots(1,1, figsize=(2,2))

    palette = {'CW': '#fe02a2', 'CCW': SCENE_COLOR}
    markers = {'CW': mmarkers.MarkerStyle('+'), 'CCW': mmarkers.MarkerStyle('x')}

    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', zorder=-1)
    sns.scatterplot(data=df, x=TRUE_ROTATION, y=DECODED_ROTATION, hue='rotation_dir', style='rotation_dir', markers=markers, ax=ax, s=10, clip_on=False, palette=palette)
    ax.axis('square')
    ax.set_xticks([-1.5*np.pi,-np.pi,-np.pi/2, 0, np.pi/2, np.pi, 1.5*np.pi])
    ax.set_xticklabels([-270, -180, -90, 0, 90, 180, 270])
    ax.set_yticks([-1.5*np.pi,-np.pi,-np.pi/2, 0, np.pi/2, np.pi, 1.5*np.pi])
    ax.set_yticklabels([-270, -180, -90, 0, 90, 180, 270])
    ax.set_xlim(-1.5*np.pi, 1.5*np.pi)
    ax.set_ylim(-1.5*np.pi, 1.5*np.pi)
    ax.set_xlabel('Actual {} [deg]'.format(TRUE_ROTATION.replace("_", " ")))
    ax.set_ylabel('{} [deg]'.format(DECODED_ROTATION.replace("_", " ")))
    ax.legend()
    fig.savefig(os.path.join(figpath, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def scatter_decoded_rotation_vs_platform_velocity(df, aggregate_by='animal', fname='motor_rotations.pdf', figpath='.', fit=True, T=3, align_rotations=False):

    sns.set_context(rc={"font.size": SMALL_SIZE, "axes.labelsize": MEDIUM_SIZE,
                        "axes.labelsize": MEDIUM_SIZE, "xtick.labelsize": SMALL_SIZE,
                        "ytick.labelsize": SMALL_SIZE})

    if align_rotations:
        df = align_rotations(df, 'motor_velocity')

    fig, ax = plt.subplots(1,1, figsize=(2.5,2.0))
    vels = np.unique(df['motor_velocity'])

    x = df['motor_velocity']
    x += 5 * np.random.randn(len(x))

    ax.scatter(x, df[DECODED_ROTATION], c='k', alpha=0.33, s=1, zorder=0, clip_on=False)

    ax.set_xticks(vels)
    ax.set_xlabel('Peak {} velocity [deg/s]'.format(PLATFORM_STR.lower()))
    ax.set_ylabel('Shift [deg]')

    ax.set_xticks(vels)
    ax.set_xlim(vels[0]-20, vels[-1]+20)
    if align_rotations:
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels([0, 90, 180, 270, 360])
        ax.set_ylim(-np.pi/4, 2*np.pi+np.pi/4)
    else:
        ax.set_yticks([-2*np.pi,-np.pi,0, np.pi, 2*np.pi])
        ax.set_yticklabels([-360, -180, 0, 180, 360])
        ax.set_ylim(-2*np.pi-np.pi/4, 2*np.pi+np.pi/4)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticks([-135, -45, 45, 135], minor=True)

    if align_rotations:
        ax.plot(45*np.arange(0,5), 45*np.arange(0,5)*T/90, c=HEAD_COLOR, marker='.', markersize=5, markeredgewidth=0.5, markeredgecolor= 'white', zorder=2, label='Actual HD')
    else:
        ax.plot(45*np.arange(-4,5), 45*np.arange(-4,5)*T/90, c=HEAD_COLOR, marker='.', markersize=5, markeredgewidth=0.5, markeredgecolor= 'white', zorder=2, label='Actual HD')

    if fit:
        slope = comp_prop_factor(df, 'motor_velocity', DECODED_ROTATION)
        r2 = r2_score(df[DECODED_ROTATION].to_numpy(), slope * df['motor_velocity'].to_numpy())

        if align_rotations:
            ax.plot([0, 180], [0, slope * 180], '-', c=DECODING_COLOR, zorder=3, alpha=1.0, label=DECODED_DIRECTION.replace('_', ' '))
        else:
            ax.plot([-180, 180], [-slope * 180, slope * 180], '-', c=DECODING_COLOR, zorder=3, alpha=1.0, label=DECODED_DIRECTION.replace('_', ' '))

        ax.text(-150, 1.55*np.pi, 'rel. slope = {:.2f} \n R2 = {:.2f}'.format(slope * 90 / T, r2), c=DECODING_COLOR, fontsize=MEDIUM_SIZE)

    plt.legend(bbox_to_anchor=(0.5, 0.35), loc='upper left', fontsize=8)
    ax.set_aspect(90/T)

    fig.savefig(os.path.join(figpath, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_drift(t, delta, stage, plot_transition=True, fname='drift.pdf', fig_path='.'):

    filtered_cos = ndimage.median_filter(np.cos(delta), size=80)
    filtered_sin = ndimage.median_filter(np.sin(delta), size=80)
    filtered = np.arctan2(filtered_sin, filtered_cos)

    filtered[np.hstack((np.abs(np.diff(filtered)) > np.pi, False))] = np.nan

    plt.figure(figsize=(8,2))
    plt.scatter(t, delta, s=0.5, alpha=0.5)
    plt.plot(t, filtered, 'k')
    plt.xlim(t[0], t[-1])
    plt.ylim([-np.pi, np.pi])
    plt.xticks(300 * np.arange(np.ceil(t[-1] / 300 + 1)), 5 * np.arange(np.ceil(t[-1] / 300) + 1).astype('int'))
    plt.hlines(0, t[0], t[-1], 'r', alpha=0.5)
    plt.xlabel('time [min]')
    plt.ylabel('deviation [rad]')
    if plot_transition:
        plt.vlines(np.min(t[stage==1]), -np.pi, np.pi, 'k', linestyle='--', alpha=0.5)
    plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']);
    plt.savefig(os.path.join(fig_path, fname), bbox_inches='tight')
    plt.close()


def scatter_decoded_vs_head_rotation_in_intervals(df, fname='transient_shifts.pdf', figpath='.', fit=False):

    fig, ax1 = plt.subplots(1,1, figsize=(1.5,1.5))

    mask = np.abs(df[TRUE_ROTATION]) > np.pi/5 #np.logical_and(np.abs(df[TRUE_ROTATION]) > np.pi/5, np.abs(df[DECODED_ROTATION]) > np.pi/5)
    df = df[mask]

    ax = sns.scatterplot(data=df, x=TRUE_ROTATION, y=DECODED_ROTATION, marker='.', s=15, zorder=1)

    ax.set_xlim(-1.5*np.pi, 1.5*np.pi)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

    ax.set_ylim(-1.5*np.pi, 1.5*np.pi)
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

    ax.set_aspect('equal')
    ax.set_xlabel('{} [rad]'.format(TRUE_ROTATION.replace("_", " ")))
    ax.set_ylabel('{} [rad]'.format(DECODED_ROTATION.replace("_", " ")))

    if fit:
        slope = comp_prop_factor(df, TRUE_ROTATION, DECODED_ROTATION)
        r2 = r2_score(df[DECODED_ROTATION].to_numpy(), slope * df[TRUE_ROTATION].to_numpy())
        ax.plot([-1.25*np.pi, 1.25*np.pi], [-slope * 1.25*np.pi, slope * 1.25*np.pi], 'r--',  zorder=2)
        ax.text(-1.1*np.pi, np.pi, 'slope = {:.2f} \n R² = {:.2f}'.format(slope, r2))

    ax.legend([],[], frameon=False)
    fig.savefig(os.path.join(figpath, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def cmap_legend(fig_path, label='Head direction', fname="hd_colorbar.pdf", vertical=True, fontsize=MEDIUM_SIZE):
    a = np.array([[0,1]])

    if vertical:
        plt.figure(figsize=(0.35,1.25))
    else:
        plt.figure(figsize=(1.25,0.25))

    img = plt.imshow(a, cmap=cmocean.cm.phase, vmin=0, vmax=2*np.pi)
    plt.gca().set_visible(False)
    #cbar = plt.colorbar(cax=cax, ticks=[0, np.pi, 2*np.pi], orientation='horizontal')
    if vertical:
        cax = plt.axes([0.05, 0.1, 0.35, 0.8])
        cbar = plt.colorbar(cax=cax, ticks=[0, np.pi, 2*np.pi], orientation='vertical')
        cbar.ax.set_yticklabels([0, 180, 360])
        cbar.set_label('{} [deg]'.format(label), fontsize=fontsize, rotation=90, labelpad=5);
    else:
        cax = plt.axes([0.1, 0.05, 0.8, 0.35])
        cbar = plt.colorbar(cax=cax, ticks=[0, np.pi, 2*np.pi], orientation='horizontal')
        cbar.ax.set_xticklabels([0, 180, 360])
        cbar.set_label('{} [deg]'.format(label), fontsize=fontsize, rotation=0, labelpad=5);  #rotation=270, labelpad=15
    cbar.set_ticklabels([0, 180, 360], fontsize=fontsize)
    plt.savefig(os.path.join(fig_path, fname), bbox_inches='tight')
    plt.close()


def dark_rotation_comparison(df_dark, style, style_order, markers, y=DECODED_ROTATION, fontsize=SMALL_SIZE, show_legend=True, fname='dark_rotation_comparison.pdf', fig_path='.'):

    fig = plt.figure(figsize=(2.5, 2.0))
    ax1 = fig.add_axes([0.2, 0.225, 0.75, 0.65])

    ax = sns.lineplot(data=df_dark, x="vel", y=y, estimator='mean', style=style, style_order=style_order,
                      errorbar='se', color=DARK_COLOR, ax=ax1, markers=markers, markeredgewidth=0.5, markeredgecolor='w')

    vels = np.unique(df_dark['vel'])
    ax.set_xlabel('Peak {} speed [deg/s]'.format(PLATFORM_STR))
    ax.set_ylabel('{} [%]'.format(DECODED_ROTATION.replace('_', ' ')))
    ax.set_ylim(-5,75)
    ax.set_yticks([0, 25, 50, 75])
    ax.set_xticks(vels)
    ax.set_xlim(35, 190)

    if show_legend:
        legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 1.225), fontsize=fontsize)
        for handle in legend.legendHandles:
            handle.set_markeredgewidth(0.5)
            handle.set_markeredgecolor('w')

    fig.savefig(os.path.join(fig_path, fname), dpi=300)
    plt.close(fig)


def plot_vel_dependence_with_regression(df, y='delta_decoded', sm_result=None, color='k', reg_color='r', txt_loc=[0.3, 0.25], pval_corr=1.0, fname='delta.pdf', fig_path='.'):

    fig = plt.figure(figsize=(2.25, 2.0))
    ax = fig.add_axes([0.225, 0.2, 0.75, 0.75])

    sns.scatterplot(data=df, x="vel", y=y, ax=ax, color=color, marker='.')

    vels = [45, 90, 135, 180]
    ax.set_xlabel('Peak velocity [deg/s]')

    if 'delta' in y:
        if y == 'delta_{}'.format(DECODED_ROTATION):
            ax.set_ylabel(r'$\Delta$ decoded rotation [%]')
        else:
            ax.set_ylabel(r'$\Delta$ neg. eye rotation [%]')
    else:
        ax.set_ylabel('{} [%]'.format(y.replace("_", " ")))

    if ax.get_ylim()[0] > 0:
        ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xticks(vels)
    ax.set_xlim(vels[0]-10, vels[-1]+10)

    if sm_result is not None:
        a = sm_result.params['const']
        b = sm_result.params['vel']
        ax.plot([45, 180], [a + b * 45, a + b * 180], reg_color)

        ax.text(txt_loc[0], txt_loc[1]+0.15, r'$y = a + b x$', fontsize=SMALL_SIZE, transform=fig.transFigure)
        ax.text(txt_loc[0]+0.3, txt_loc[1]+0.15, r'$R^2 = {:.1f}$'.format(sm_result.rsquared), fontsize=SMALL_SIZE, transform=fig.transFigure)
        ax.text(txt_loc[0],txt_loc[1]+0.075, r'$a = {:.1f}\pm{:.1f}$ ({})'.format(a, sm_result.bse['const'], convert_pvalue_str(sm_result.pvalues['const'] * pval_corr)), transform=fig.transFigure)
        ax.text(txt_loc[0],txt_loc[1], r'$b = {:.1f}\pm{:.1f}$ ({})'.format(b, sm_result.bse['vel'], convert_pvalue_str(sm_result.pvalues['vel'] * pval_corr)), transform=fig.transFigure)

    fig.savefig(os.path.join(fig_path, fname), dpi=300)
    plt.close(fig)


def loss_heatmap(p1, p2, loss, p1fit=None, p2fit=None, p1txt='p1', p2txt='p2', fname='loss_heatmap.pdf', fig_path='.'):
    fig = plt.figure(figsize=(2.5,1.75))
    ax = fig.add_subplot(111)
    aspect = p1[-1] / p2[-1]
    cax = ax.matshow(np.log(loss).T, cmap='inferno_r', aspect=aspect, extent=[p1[0], p1[-1], p2[0], p2[-1]], origin='lower')
    ax.set_xlabel(p1txt)
    ax.set_ylabel(p2txt)
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)

    if p1fit is not None:
        ax.scatter([p1fit], [p2fit], s=3)

    colorbar = plt.colorbar(cax)
    colorbar.set_label('log loss')
    plt.subplots_adjust(right=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_path, fname), dpi=300)
    plt.close(fig)


def scatter_angular_variables(x, y, rho=None, xlabel='Decoded HD [deg]', ylabel='Actual HD [deg]', fname='scatterplot_angular_variables.pdf', fig_path='.'):
    fig, ax = plt.subplots(1,1, figsize=(1.5,1.5))
    ax.scatter(180 * x / np.pi, 180 * y / np.pi, s=0.15, alpha=0.25, c='k')
    ax.plot([0, 360], [0, 360], 'r--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 180, 360])
    ax.set_yticks([0, 180, 360])
    ax.axis('equal')
    ax.set_xlim(0,360)
    ax.set_ylim(0,360)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    if rho:
        ax.text(30, 320, r"$\rho = {:1.2f}$".format(rho), fontsize=SMALL_SIZE)
    fig.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_rotation_over_time_dark(df, T=3, offset=1, color='k', fname='rotation_time_series_dark.pdf', figpath='.'):

    fig = plt.figure(figsize=(1.75,1.3))
    ax = fig.add_axes([0.4,.28,.6,.65])

    df['rotation_deg'] = df[DECODED_ROTATION] * 180 / np.pi
    ax1 = sns.lineplot(data=df, x='t', y='rotation_deg', style='phenotype', style_order=['WT', 'FRMD7'],
                       legend=False, color=color, estimator='mean', errorbar='se', markers=False, ax=ax)

    ax.set_xlabel('t [s]')
    ax.set_ylabel('Decoded head\ndirection [deg]')
    ax.set_xlim(-offset, T+offset)
    ax.axvline(0, color='k', linestyle='--', zorder=-1)
    ax.axvline(T, color='k', linestyle='--', zorder=-1)
    ax.set_xticks(np.arange(T+1))

    fig.savefig(os.path.join(figpath, fname), dpi=300)
    plt.close(fig)
