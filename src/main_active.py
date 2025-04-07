from vis_vest_hd_computation.imports import *
import vis_vest_hd_computation.analyze_recording as analyze_recording
import vis_vest_hd_computation.analyze_active as analyze_active
import vis_vest_hd_computation.visualization as visualization
import vis_vest_hd_computation.utils as utils
import itertools
from itertools import compress
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score


class MaskableList(list):
    def __getitem__(self, index):
        try: return super(MaskableList, self).__getitem__(index)
        except TypeError: return MaskableList(compress(self, index))


def main(recording_type, frmd7=False, demo=False):

    eyetracking = False
    dark = False

    if frmd7:
        data_file = os.path.join(BASE_PATH, 'data', 'FRMD7_active.csv')
    else:
        data_file = os.path.join(BASE_PATH, 'data', 'WT_active.csv')

    fig_path = utils.get_fig_path(BASE_PATH, recording_type, eyetracking, frmd7)

    df = pd.read_csv(data_file, index_col=0)
    animals = list(df.index)
    n_animals = len(animals)

    n_stages = {'active_3landmarks': 1, 'combined': 2, 'active_grating_stim': 1}
    decoding_error = np.nan * np.zeros((n_animals, n_stages[recording_type]))
    mean_vec_length = np.nan * np.zeros((n_animals, n_stages[recording_type]))
    mean_latent_vec_length = np.nan * np.zeros((n_animals, n_stages[recording_type]))
    n_cells = np.nan * np.zeros((n_animals, n_stages[recording_type]), dtype='int')

    n_lags = 2 * FPS_MINISCOPE
    lag = np.arange(-n_lags, n_lags + 1) / FPS_MINISCOPE
    circ_corr = np.nan * np.zeros((n_animals, n_stages[recording_type], 2 * n_lags + 1))

    vec_length = []
    latent_vec_length = []
    significant_hd_tuning = []
    significant_latent_tuning = []
    peak_firing_dir = []
    latent_peak_firing_dir = []

    if recording_type == 'combined':
        category_type = 'stage'
        category1 = 'active \n + landmarks'
        category2 = 'passive \n + grating'
        alternative1 = 'less'
        alternative2 = 'greater'
    elif recording_type == 'active_grating_stim':
        category_type = 'rotation_dir'
        category1 = 'CW'
        category2 = 'CCW'
        alternative = 'greater'

    combined_recordings = []
    active_grating_stim_recordings = []

    for k, a in enumerate(animals):

        recording = df.at[a, recording_type]

        print(recording)

        if not pd.isna(recording):
            res = analyze_active.main(recording, recording_type)
            decoding_error[k] = res['decoding_error']
            mean_vec_length[k] = [np.mean(x) for x in res['vec_length']]
            mean_latent_vec_length[k] = [np.mean(x) for x in res['latent_vec_length']]
            n_cells[k] = res['n_cells']
            circ_corr[k] = res['circ_corr']

            vec_length.append(res['vec_length'])
            latent_vec_length.append(res['latent_vec_length'])
            significant_hd_tuning.append(res['significant_hd_tuning'])
            significant_latent_tuning.append(res['significant_latent_tuning'])
            peak_firing_dir.append(res['peak_firing_dir'])
            latent_peak_firing_dir.append(res['latent_peak_firing_dir'])

            if recording_type == 'combined':
                analyze_recording.main(a, recording, eyetracking, stage=1, dark=dark)
                combined_recordings.append(recording)
            elif recording_type == 'active_grating_stim':
                analyze_recording.main(a, recording, eyetracking, stage=0, dark=dark)
                active_grating_stim_recordings.append(recording)

    animals = MaskableList(animals)

    if n_stages[recording_type] == 2 and not demo:
        mad_1, mad_2, mask = extract_stages_from_arr(decoding_error * 180 / np.pi)
        stat, p_value = ttest_rel(mad_1, mad_2, alternative=alternative1)

        df_mad = create_df(animals[mask], mad_1, mad_2, statistic='decoding_error')
        fname = 'decoding_error.pdf'
        visualization.plot_statistic(df_mad, statistic='decoding_error', unit='deg', p_value=p_value, fname=fname, fig_path=fig_path, category1=category1.capitalize(), category2=category2.capitalize(), category_type=category_type, category_fontsize=6) #width=2.25, sizes=15,

        vec_length_1, vec_length_2, mask = extract_stages_from_arr(mean_vec_length)
        stat, p_value = ttest_rel(vec_length_1, vec_length_2, alternative=alternative2)
        df_vec_length = create_df(animals[mask], vec_length_1, vec_length_2, statistic='vec_length')
        fname = 'vec_length.pdf'
        visualization.plot_statistic(df_vec_length, statistic='vec_length', p_value=p_value, fname=fname, fig_path=fig_path, category1=category1.capitalize(), category2=category2.capitalize(), category_type=category_type, category_fontsize=6)

        latent_vec_length_1, latent_vec_length_2, mask = extract_stages_from_arr(mean_latent_vec_length)
        stat, p_value = ttest_rel(latent_vec_length_1, latent_vec_length_2)
        df_latent_vec_length = create_df(animals[mask], latent_vec_length_1, latent_vec_length_2, statistic='vec_length')
        fname = 'latent_vec_length.pdf'
        visualization.plot_statistic(df_latent_vec_length, statistic='vec_length', p_value=p_value, fname=fname, fig_path=fig_path, category1=category1.capitalize(), category2=category2.capitalize(), category_type=category_type, category_fontsize=6)

        stat, p_value = ttest_rel(vec_length_1, latent_vec_length_1)
        df_latent_vec_length = create_df(animals[mask], vec_length_1, latent_vec_length_1, statistic='vec_length')
        fname = 'hd_vs_latent_vec_length.pdf'
        visualization.plot_statistic(df_latent_vec_length, statistic='vec_length', p_value=p_value, fname=fname, fig_path=fig_path, category1='Actual', category2='Decoded', category_type=category_type, category_fontsize=6)

    if not demo:
        for stage in range(n_stages[recording_type]):

            perc_tuned_cells = [np.mean(x[stage]) for x in significant_hd_tuning]

            # mean_tuned_cells = np.mean(perc_tuned_cells)
            # std_tuned_cells = np.std(perc_tuned_cells, ddof=1)

            # mean_cells = np.mean(n_cells)
            # std_cells = np.std(n_cells[:,0], ddof=1)

            ymax = 100
            yticks = [0, 25, 50, 75, 100]
            ylim = [0, 100]
            visualization.mean_std_scatter(100 * np.array(perc_tuned_cells), fname='percentage_tuned.pdf', label='% tuned', lim=ylim, ticks=yticks, fig_path=fig_path)

            fname = 'vec_length_histogram_stage={}.pdf'.format(stage)
            visualization.vec_length_hist(vec_length, significant_hd_tuning, stage, frmd7=frmd7, fname=fname, figpath=fig_path)

            fname = 'latent_vec_length_histogram_stage={}.pdf'.format(stage)
            visualization.vec_length_hist(latent_vec_length, significant_latent_tuning, stage, frmd7=frmd7, fname=fname, figpath=fig_path)

        nticks = int(np.ceil(np.max(decoding_error[:,0]) * 180 / (np.pi * 10)))
        yticks = 10 * np.arange(nticks + 1)
        ymax = 10 * nticks
        for horizontal in [True, False]:
            fname = 'decoding_error_stage=0_horizontal={}.pdf'.format(horizontal)
            print(decoding_error[:,0] * 180 / np.pi)
            visualization.mean_std_scatter(decoding_error[:,0] * 180 / np.pi, fname=fname, label='Decoding error [deg]', lim=[0, ymax], ticks=yticks, fig_path=fig_path, horizontal=horizontal)

        pop_size = n_cells[:,0].squeeze()
        ymax = int(np.ceil(np.nanmax(pop_size) / 50))
        yticks = 50 * np.arange(ymax+1)
        ylim = [0, 50 * ymax]
        visualization.mean_std_scatter(n_cells[:,0].squeeze(), fname='number_of_neurons.pdf', label='# neurons', lim=ylim, ticks=yticks, fig_path=fig_path)

        if recording_type == 'combined' or recording_type == 'active_3landmarks':
            visualization.plot_circ_corr(lag, circ_corr[:,0,:].squeeze(), fig_path=fig_path)

    if recording_type == 'combined':

        df_all_comb = load_data(combined_recordings, 'agg_data')

        visualization.scatter_decoded_rotation_vs_platform_velocity(df_all_comb, figpath=fig_path)

        if not demo:

            latent_vec_length_1 = extract_stage(latent_vec_length, 0)
            latent_vec_length_2 = extract_stage(latent_vec_length, 1)

            fname = 'latent_vec_length_scatter.pdf'
            visualization.scatter_vec_length(latent_vec_length_1, latent_vec_length_2, xlabel='VL, active + landmarks', ylabel='VL, passive + grating', fname=fname, figpath=fig_path)

            vec_length_1 = extract_stage(vec_length, 0)
            vec_length_2 = extract_stage(vec_length, 1)

            fname = 'hd_vec_length_scatter.pdf'
            visualization.scatter_vec_length(vec_length_1, vec_length_2, xlabel='VL, active + landmarks', ylabel='VL, passive + grating', fname=fname, figpath=fig_path)

            fname = 'hd_vs_latent_vec_length_scatter.pdf'
            visualization.scatter_vec_length(vec_length_1, latent_vec_length_1, xlabel='Actual VL', ylabel='Decoded VL', fname=fname, figpath=fig_path)

            # from astropy.stats import rayleightest
            # pvals = [rayleightest(x[0]) for x in peak_firing_dir]
            # print(pvals)

            peak_firing_dir_1 = extract_stage(peak_firing_dir, 0)
            latent_peak_firing_dir_1 = extract_stage(latent_peak_firing_dir, 0)
            latent_peak_firing_dir_2 = extract_stage(latent_peak_firing_dir, 1)

            peak_firing_dir_1 = utils.map_angle(peak_firing_dir_1)
            latent_peak_firing_dir_1 = utils.map_angle(latent_peak_firing_dir_1)
            latent_peak_firing_dir_2 = utils.map_angle(latent_peak_firing_dir_2)

            fname = 'phd_hist2d_hd_vs_latent.pdf'
            visualization.pfd_hist2d(peak_firing_dir_1, latent_peak_firing_dir_2, xlabel='Actual PHD [deg]', ylabel='Decoded PHD [deg]', fname=fname, fig_path=fig_path)

            fname = 'phd_hist2d_latent_active_vs_passive.pdf'
            visualization.pfd_hist2d(latent_peak_firing_dir_1, latent_peak_firing_dir_2, xlabel='PHD, active + landmarks [deg]', ylabel='PHD, passive + grating [deg]', fname=fname, fig_path=fig_path)

            delta_pfd_latent_hd = latent_peak_firing_dir_1 - peak_firing_dir_1
            delta_pfd_latent_hd = np.angle(np.exp(1j * delta_pfd_latent_hd))

            delta_latent_pdf_passive_active = latent_peak_firing_dir_2 - latent_peak_firing_dir_1
            delta_latent_pdf_passive_active = np.angle(np.exp(1j * delta_latent_pdf_passive_active))

            fname = 'delta_pfd_latent_hd.pdf'
            visualization.delta_pfd_hist(delta_pfd_latent_hd, fname=fname, figpath=fig_path)

            fname = 'delta_latent_pdf_passive_active.pdf'
            visualization.delta_pfd_hist(delta_latent_pdf_passive_active, fname=fname, figpath=fig_path)

            slopes = []
            r2s = []
            for animal in df_all_comb['animal'].unique():
                df = df_all_comb[df_all_comb['animal']==animal]
                slope = utils.comp_prop_factor(df, 'motor_velocity', DECODED_ROTATION)
                r2 = r2_score(df[DECODED_ROTATION].to_numpy(), slope * df['motor_velocity'].to_numpy())
                slopes.append(slope * 90 / T_TRANSIENT_STIM)
                r2s.append(r2)

            visualization.mean_std_scatter(slopes, fname='rel_slope_animals.pdf', label='rel. slope', lim=[0.7, 1], ticks=[0.7,0.8,0.9,1.0], fig_path=fig_path)
            visualization.mean_std_scatter(slopes, fname='r2_animals.pdf', label='R2', lim=[0, 1], ticks=[0,0.5,1], fig_path=fig_path)

    if recording_type == 'active_grating_stim':

        df_active_grating_stim = load_data(active_grating_stim_recordings, 'active_grating_data')
        rot_dir = ['CCW' if v > 0 else 'CW' for v in df_active_grating_stim['grating_velocity']]
        df_active_grating_stim['rotation_dir'] = rot_dir
        visualization.scatter_decoded_vs_head_rotation(df_active_grating_stim, figpath=fig_path)

        if not demo:
            df_active_grating_stim = df_active_grating_stim.groupby(['animal', 'rotation_dir'])['delta_rotation'].agg('mean').reset_index()
            df_active_grating_stim.sort_values(by=['animal', 'rotation_dir'])
            df_active_grating_stim['delta_rotation'] *= 180 / np.pi

            mask_cw = df_active_grating_stim['rotation_dir'] == 'CW'
            delta_rotation_cw = df_active_grating_stim['delta_rotation'][mask_cw].to_numpy()

            mask_ccw = df_active_grating_stim['rotation_dir'] == 'CCW'
            delta_rotation_ccw = df_active_grating_stim['delta_rotation'][mask_ccw].to_numpy()

            stat, p_value = ttest_rel(delta_rotation_cw, delta_rotation_ccw, alternative=alternative)

            fname = 'delta_rotation.pdf'
            visualization.plot_statistic(df_active_grating_stim, statistic='delta_rotation', unit='deg', p_value=p_value, fname=fname, fig_path=fig_path, category1=category1, category2=category2, category_type=category_type)


def extract_stages_from_arr(x):
    x1 = x[:,0]
    x2 = x[:,1]
    mask = np.logical_not(np.isnan(x1))
    x1 = x1[mask]
    x2 = x2[mask]
    return x1, x2, mask


def create_df(animals, x1, x2, statistic='statistic'):
    n = len(animals)
    df = pd.DataFrame()
    df['animal'] = animals * 2
    df['stage'] = np.hstack((np.zeros(n, dtype=int), np.ones(n, dtype=int)))
    df[statistic] = np.hstack((x1, x2))
    return df


def extract_stage(x, k):
    out = [y[k] for y in x]
    out = list(itertools.chain(*out))
    return np.array(out)


def load_data(recordings, data_type):
    df_list = []
    for recording in recordings:
        df_recording = pd.read_pickle(os.path.join(BASE_PATH, 'data', 'trial_data', recording, '{}.p'.format(data_type)))
        df_recording['recording'] = recording
        df_recording['animal'] = recording[:6]
        df_list.append(df_recording)
    df = pd.concat(df_list, ignore_index=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--recording_type', type=str, required=True)
    parser.add_argument('-f', '--frmd7', action='store_true')
    parser.add_argument('-d', '--demo', action='store_true')
    args = parser.parse_args()

    main(args.recording_type, args.frmd7, demo=args.demo)
