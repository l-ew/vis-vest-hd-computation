from sklearn.metrics import mean_squared_error
import vis_vest_hd_computation.model as model
import vis_vest_hd_computation.utils as utils
import vis_vest_hd_computation.visualization as visualization
import vis_vest_hd_computation.eye_motion_regression as eye_motion_regression
from vis_vest_hd_computation.linear_comb import linear_comb
from vis_vest_hd_computation.imports import *


def main(animals, stimset, eyetracking=False, include_dark=False, include_asymmetric=False, frmd7=False, T=T_TRANSIENT_STIM):

    data_dir = utils.get_data_str(stimset, include_dark)
    data_path = os.path.join(BASE_PATH, 'data', 'trial_data')
    out_path = utils.get_out_path(BASE_PATH, data_dir, eyetracking, frmd7, include_asymmetric=include_asymmetric)
    fig_path = utils.get_fig_path(BASE_PATH, data_dir, eyetracking, frmd7, include_asymmetric=include_asymmetric)

    df_restricted = utils.load_data(data_path, animals, data_dir=data_dir, agg_level='animals', data_type='restricted_stimset')

    visualization.plot_restricted(df_restricted, var=DECODED_ROTATION, large_fig=True, fig_path=fig_path)
    if eyetracking:
        visualization.plot_restricted(df_restricted, var=NEG_EYE_ROTATION, fig_path=fig_path)
        visualization.scatter_decoded_rotation_vs_negative_eye_rotation_restricted(df_restricted[~df_restricted['dark']], fig_path=fig_path)

    df_fname = 'restricted_stimset.p'
    df_restricted.to_pickle(os.path.join(out_path, df_fname))

    linear_comb(df_restricted, fname='linear_combination.pdf', fig_path=fig_path)

    df = utils.load_data(data_path, animals, data_dir=data_dir, agg_level='animals', data_type='agg_data')

    pklname = 'agg_data.p'
    df.to_pickle(os.path.join(out_path, pklname))

    df_bright = df[~df['dark']].copy()
    visualization.plot_all_combinations(df_bright, var=DECODED_ROTATION, large_fig=True, fig_path=fig_path)

    data_trials = utils.load_all_combinations_data(data_path, data_dir, out_path, animals)
    decoded_rot_time_series = utils.get_rotation_time_series(data_trials, agg_dirs=True, dark=False)
    mean_decoded_rot_time_series = utils.avg_data(decoded_rot_time_series, var=DECODED_ROTATION)
    mean_decoded_rot_trimmed_time_series = utils.trim_data(mean_decoded_rot_time_series, t_post=0)

    if eyetracking and not frmd7:
        visualization.plot_all_combinations(df_bright, var=NEG_EYE_ROTATION, fig_path=fig_path)

        fig_path_decoded_vs_negative_eye_rotation = os.path.join(fig_path, 'decoded_vs_negative_eye_rotation')
        if not os.path.exists(fig_path_decoded_vs_negative_eye_rotation):
            os.makedirs(fig_path_decoded_vs_negative_eye_rotation)
        visualization.scatterplot_all_combinations(df_bright, fig_path=fig_path_decoded_vs_negative_eye_rotation)

        df_eye_vel, df_eye_pos = utils.comp_eye_time_series(df_bright)
        df_eye_vel = utils.align_rotations(df_eye_vel, 'motor_velocity')
        df_eye_pos = utils.align_rotations(df_eye_pos, 'motor_velocity')
        df_mean_eye_vel = utils.avg_data(df_eye_vel, var=EYE_VELOCITY)
        df_mean_eye_pos = utils.avg_data(df_eye_pos, var=[EYE_ROTATION, NEG_EYE_ROTATION])

        df_combined = utils.combine_data(mean_decoded_rot_trimmed_time_series, df_mean_eye_vel, df_mean_eye_pos)
        plant_lag, plant_coef = eye_motion_regression.regression(df_combined, out_path=out_path, fig_path=fig_path)

        fit_gamma_r = True
        fit_gamma_c = True
        model_params, loss_fun, sample_weight_vel, sample_weight_dir = model.fit_model(df_combined, df_dark=None, loss_type='eye_velocity', fit_gamma_r=fit_gamma_r, fit_gamma_c=fit_gamma_c, plant_lag=plant_lag, plant_coef=plant_coef, norm=True, t_post=0)
        np.savetxt(os.path.join(out_path, "model_params.csv"), model_params, delimiter=",")

        if fit_gamma_c and fit_gamma_r:
            fname = 'sensitivity_analysis_visual_gain.pdf'
            sensitivity_analysis(loss_fun, model_params, k1=1, k2=3, p1max=2, p2max=1, p1txt=r'$g_{r0}$', p2txt=r'$\gamma_r$', fname=fname, fig_path=fig_path)

            fname = 'sensitivity_analysis_vestibular_gain.pdf'
            sensitivity_analysis(loss_fun, model_params, k1=2, k2=4, p1txt=r'$g_{c0}$', p2txt=r'$\gamma_c$', fname=fname, fig_path=fig_path)

        utils.align_rotations(df_bright, 'motor_velocity')
        df_bright = utils.avg_data(df_bright, var=[DECODED_ROTATION, NEG_EYE_ROTATION])
        df_bright = df_bright.sort_values(by=['motor_velocity', 'grating_velocity', 'static_grating_interval'])

        df_model_est_rotation = df_bright[['motor_velocity', 'grating_velocity', 'static_grating_interval', DECODED_ROTATION, NEG_EYE_ROTATION]].copy()
        df_model_est_rotation, df_model_est_rotation_time_series, df_model_eye_vel, df_model_eye_pos = model.model_pred(df_model_est_rotation, model_params, fit_gamma_r=fit_gamma_r, fit_gamma_c=fit_gamma_c, plant_lag=plant_lag, plant_coef=plant_coef, t_post=0)

        fig_path_model_vs_decoded_rotation = os.path.join(fig_path, 'model_vs_decoded_rotation')
        if not os.path.exists(fig_path_model_vs_decoded_rotation):
            os.makedirs(fig_path_model_vs_decoded_rotation)
        fname = 'model_vs_decoded_rotation_all_combinations.pdf'
        visualization.scatterplot_all_combinations(df_model_est_rotation, xcol=DECODED_ROTATION, ycol=MODEL_ROTATION, is_aligned=True, is_avg=True, fname=fname, fig_path=fig_path_model_vs_decoded_rotation)

        df_model_restricted = utils.extract_restricted_data(df_model_est_rotation)
        visualization.plot_restricted(df_model_restricted, var=MODEL_ROTATION, fig_path=fig_path)
        visualization.plot_restricted(df_model_restricted, var=MODEL_NEG_EYE_ROTATION, fig_path=fig_path)
        visualization.plot_all_combinations(df_model_est_rotation, var=MODEL_ROTATION, fig_path=fig_path)
        visualization.plot_all_combinations(df_model_est_rotation, var=MODEL_NEG_EYE_ROTATION, fig_path=fig_path)

        if include_dark:
            df_dark = df[df['dark']].copy()
            utils.align_rotations(df_dark, 'motor_velocity')

            df_dark = utils.avg_data(df_dark, var=[DECODED_ROTATION, NEG_EYE_ROTATION])
            df_dark = df_dark.sort_values(by=['motor_velocity', 'grating_velocity', 'static_grating_interval'])
            df_dark = df_dark[['motor_velocity', 'grating_velocity', 'static_grating_interval', DECODED_ROTATION, NEG_EYE_ROTATION]].copy()

            df_model_est_rotation_dark, df_model_est_rotation_time_series_dark, _, _ = model.model_pred(df_dark.copy(), model_params, fit_gamma_r=fit_gamma_r, fit_gamma_c=fit_gamma_c, plant_lag=plant_lag, plant_coef=plant_coef, dark=True, t_post=0)
            df_model_restricted_dark = utils.extract_restricted_data(df_model_est_rotation_dark, dark=True)
            df_model_restricted = pd.concat([df_model_restricted, df_model_restricted_dark], ignore_index=True)

            df_model_est_rotation_dark_adj, df_model_est_rotation_time_series_dark_adj = model.adjust_vestibular_gain_in_darkness(df_dark.copy(), model_params, fit_gamma_r=fit_gamma_r, fit_gamma_c=fit_gamma_c, plant_lag=plant_lag, plant_coef=plant_coef, t_post=0)
            df_model_est_rotation_dark['{}_adj'.format(MODEL_ROTATION)] = df_model_est_rotation_dark_adj[MODEL_ROTATION]
            df_model_est_rotation_time_series_dark['{}_adj'.format(MODEL_ROTATION)] = df_model_est_rotation_time_series_dark_adj[MODEL_ROTATION]

            utils.normalize_restricted(df_model_restricted_dark)
            df_dark_combined = pd.melt(df_model_restricted_dark, var_name='rotation_type', value_name='shift', id_vars=['vel'], value_vars=[DECODED_ROTATION, MODEL_ROTATION])
            df_dark_combined.loc[df_dark_combined['rotation_type']==MODEL_ROTATION, 'rotation_type'] = 'Model'
            df_dark_combined.loc[df_dark_combined['rotation_type']==DECODED_ROTATION, 'rotation_type'] = 'Data'

            df_model_restricted_dark_adj = utils.extract_restricted_data(df_model_est_rotation_dark_adj, dark=True)
            utils.normalize_restricted(df_model_restricted_dark_adj)
            df_adj = df_model_restricted_dark_adj[['vel', MODEL_ROTATION]].copy()
            df_adj['rotation_type'] = 'Adjusted model'
            df_adj = df_adj.rename({MODEL_ROTATION: 'shift'}, axis='columns')
            df_dark_combined = pd.concat([df_dark_combined, df_adj], ignore_index=True)

            import matplotlib.markers as mmarkers
            style = 'rotation_type'
            style_order = ['Data', 'Model', 'Adjusted model']
            markers = {'Data': mmarkers.MarkerStyle('o').scaled(0.5), 'Model': mmarkers.MarkerStyle('v').scaled(0.5), 'Adjusted model': mmarkers.MarkerStyle('^').scaled(0.5)}
            visualization.dark_rotation_comparison(df_dark_combined, style, style_order, markers, y='shift', fname='dark_rotations.pdf', fig_path=os.path.join(fig_path, 'model_vs_decoded_rotation'))

    elif frmd7:
        model_params = np.genfromtxt(os.path.join(data_path, 'all_combinations_eye_WT', 'model_params.csv'))
        plant_lag = np.genfromtxt(os.path.join(data_path, 'all_combinations_eye_WT', 'eye_motion_regression', 'plant_lag.csv'))
        plant_coef = np.genfromtxt(os.path.join(data_path, 'all_combinations_eye_WT', 'eye_motion_regression', 'plant_coef.csv'))

        utils.align_rotations(df_bright, 'motor_velocity')
        df_bright = utils.avg_data(df_bright, var=[DECODED_ROTATION])
        df_bright = df_bright.sort_values(by=['motor_velocity', 'grating_velocity', 'static_grating_interval'])

        df_model_est_rotation = df_bright[['motor_velocity', 'grating_velocity', 'static_grating_interval', DECODED_ROTATION]].copy()
        df_model_est_rotation, _, _, _ = model.model_pred(df_model_est_rotation, model_params, fit_gamma_r=True, fit_gamma_c=True, plant_lag=plant_lag, plant_coef=plant_coef, frmd7=True, t_post=0)
        df_model_restricted = utils.extract_restricted_data(df_model_est_rotation, dark=False)

        visualization.plot_restricted_model_frmd7(df_model_restricted, fig_path=fig_path)
        visualization.plot_all_combinations(df_model_est_rotation, var=MODEL_ROTATION, large_fig=True, fig_path=fig_path)

    decoded_rot_time_series = utils.avg_data_per_animal(decoded_rot_time_series, var=DECODED_ROTATION)

    if include_dark:
        data_trials_dark = utils.load_dark_data(data_path, data_dir, out_path, animals)
        decoded_rot_time_series_dark = utils.get_rotation_time_series(data_trials_dark, agg_dirs=True, dark=True)
        decoded_rot_time_series_dark = utils.avg_data_per_animal(decoded_rot_time_series_dark, var=DECODED_ROTATION)
        decoded_rot_time_series_dark.to_pickle(os.path.join(out_path, 'decoded_rot_time_series_dark.p'))

    if eyetracking:
        fig_path_time_series = os.path.join(fig_path, 'time_series_with_model_predictions')
        os.makedirs(fig_path_time_series, exist_ok=True)

        df_eye_pos = utils.avg_data_per_animal(df_eye_pos, var=NEG_EYE_ROTATION)

        df_eye_vel = utils.avg_data_per_animal(df_eye_vel, var=EYE_VELOCITY)
        cols = ['animal', 'motor_velocity', 'grating_velocity', 'static_grating_interval']
        df_eye_vel = df_eye_vel.groupby(cols)[EYE_VELOCITY].agg('mean').reset_index()

        fig_path_time_series_bright = os.path.join(fig_path_time_series, 'bright')
        os.makedirs(fig_path_time_series_bright, exist_ok=True)
        visualization.plot_rotation_time_series_all_stim(decoded_rot_time_series, df_model_est_rotation_time_series, df_eye_pos, df_model_eye_pos, dark=False, fig_path=fig_path_time_series_bright)

        df_eye_pos = utils.align_and_avg_scene_rotations(df_eye_pos)
        df_eye_vel = utils.align_and_avg_scene_rotations(df_eye_vel)
        decoded_rot_time_series = utils.align_and_avg_scene_rotations(decoded_rot_time_series)

        fig_path_time_series_bright = os.path.join(fig_path_time_series, 'bright_with_labels_and_aligned_scene_rotations')
        os.makedirs(fig_path_time_series_bright, exist_ok=True)
        visualization.plot_rotation_time_series_all_stim(decoded_rot_time_series, df_model_est_rotation_time_series, df_eye_pos, df_model_eye_pos, dark=False, txt=True, fig_path=fig_path_time_series_bright)

        if include_dark:
            fig_path_time_series_dark = os.path.join(fig_path_time_series, 'dark')
            os.makedirs(fig_path_time_series_dark, exist_ok=True)
            visualization.plot_rotation_time_series_all_stim(decoded_rot_time_series_dark, df_model_est_rotation_time_series_dark, None, None, dark=True, fig_path=fig_path_time_series_dark)

    df_restricted = utils.align_restricted(df_restricted)
    df_restricted = df_restricted.groupby(['animal', 'vel', 'stim_type', 'dark'])[[DECODED_ROTATION, NEG_EYE_ROTATION]].agg('mean').reset_index()

    regression_analyses(df_restricted.copy(), eyetracking=eyetracking, frmd7=frmd7, fig_path=fig_path)

    # plot times series of decoded HD shift for rotations in darkness
    fig_path_rot_time_series = os.path.join(fig_path, '{}_time_series'.format(DECODED_ROTATION))
    os.makedirs(fig_path_rot_time_series, exist_ok=True)

    if include_dark:
        visualization.plot_rotation_time_series(decoded_rot_time_series_dark, color=DARK_COLOR, cond='dark', figpath=fig_path_rot_time_series)

    # plot times series of decoded HD shift for rotations in bright conditions
    decoded_rot_time_series = utils.get_rotation_time_series(data_trials, agg_dirs=True, dark=False)
    decoded_rot_time_series = utils.avg_data_per_animal(decoded_rot_time_series, var=DECODED_ROTATION)

    for vm in [45, 90, 135, 180]:
        mask = np.logical_and(decoded_rot_time_series['motor_velocity'] == vm, decoded_rot_time_series['grating_velocity'] >= 0)
        df_decoded_rot_motor_vel = decoded_rot_time_series[mask].copy()
        visualization.plot_rotation_time_series(df_decoded_rot_motor_vel, style='grating_velocity', palette=VEL_PALETTE, dashes=False, cond='vm={}'.format(vm), figpath=fig_path_rot_time_series)

    df_decoded_rot_natural = decoded_rot_time_series[decoded_rot_time_series['grating_velocity'] == 0].copy()
    visualization.plot_rotation_time_series(df_decoded_rot_natural, color=NATURAL_COLOR, cond='natural', figpath=fig_path_rot_time_series)

    df_decoded_rot_sync = decoded_rot_time_series[decoded_rot_time_series['motor_velocity']==decoded_rot_time_series['grating_velocity']].copy()
    visualization.plot_rotation_time_series(df_decoded_rot_sync, color=SYNC_COLOR, cond='sync', figpath=fig_path_rot_time_series)

    df_decoded_rot_scene = decoded_rot_time_series[decoded_rot_time_series['motor_velocity']==0].copy()
    grating_dir = np.sign(df_decoded_rot_scene['grating_velocity'])
    df_decoded_rot_scene.loc[grating_dir < 0, 'grating_velocity'] *= -1
    df_decoded_rot_scene.loc[grating_dir > 0, DECODED_ROTATION] *= -1
    visualization.plot_rotation_time_series(df_decoded_rot_scene, style='grating_velocity', color=SCENE_COLOR, cond='scene', figpath=fig_path_rot_time_series)

    df_true_rotation = utils.get_true_rotation_time_series()
    visualization.plot_rotation_time_series(df_true_rotation, var=TRUE_ROTATION, color=STIM_COLOR, cond='true', figpath=fig_path_rot_time_series)

    # plot times series of both decoded and actual HD shift for rotations in bright conditions
    df_mean = df_restricted.groupby(['vel', 'stim_type'])[DECODED_ROTATION].agg('mean').reset_index()
    df_sem = df_restricted.groupby(['vel', 'stim_type'])[DECODED_ROTATION].agg('sem').reset_index()

    for vm in [45, 90, 135, 180]:
        df1 = df_true_rotation[df_true_rotation['motor_velocity']==vm].copy()
        df2 = df_decoded_rot_natural[df_decoded_rot_natural['motor_velocity']==vm].copy()
        df3 = df_decoded_rot_sync[df_decoded_rot_sync['motor_velocity']==vm].copy()

        natural_rotation = df_mean[np.logical_and(df_mean['stim_type']==PLATFORM_STR, df_mean['vel']==vm)][DECODED_ROTATION].iloc[0] * 180 / np.pi
        sync_rotation = df_mean[np.logical_and(df_mean['stim_type']==SYNC_STR, df_mean['vel']==vm)][DECODED_ROTATION].iloc[0] * 180 / np.pi
        true_rotation = 2 * T * vm / np.pi

        sem_natural = df_sem[np.logical_and(df_sem['stim_type']==PLATFORM_STR, df_sem['vel']==vm)][DECODED_ROTATION].iloc[0] * 180 / np.pi
        sem_sync = df_sem[np.logical_and(df_sem['stim_type']==SYNC_STR, df_sem['vel']==vm)][DECODED_ROTATION].iloc[0] * 180 / np.pi

        fig_path_natural = os.path.join(fig_path_rot_time_series, 'natural_with_actual_HD_shift')
        os.makedirs(fig_path_natural, exist_ok=True)

        fname = 'vm={}.pdf'.format(vm)
        visualization.plot_decoded_and_true_rotation_time_series(df1, df2, color=NATURAL_COLOR, true_rotation=true_rotation, decoded_rotation=natural_rotation, sem=sem_natural, fname=fname, figpath=fig_path_natural)

        fig_path_sync = os.path.join(fig_path_rot_time_series, 'sync_with_actual_HD_shift')
        os.makedirs(fig_path_sync, exist_ok=True)

        visualization.plot_decoded_and_true_rotation_time_series(df1, df3, color=SYNC_COLOR, true_rotation=true_rotation, decoded_rotation=sync_rotation, sem=sem_sync, fname=fname, figpath=fig_path_sync)

    df_decoded_rot_scene = decoded_rot_time_series[decoded_rot_time_series['motor_velocity']==0].copy()
    grating_dir = np.sign(df_decoded_rot_scene['grating_velocity'])
    df_decoded_rot_scene.loc[grating_dir < 0, 'grating_velocity'] *= -1
    df_decoded_rot_scene.loc[grating_dir > 0, DECODED_ROTATION] *= -1

    for vg in [45, 90, 135, 180]:
        df1 = df_true_rotation[df_true_rotation['motor_velocity']==vg].copy()
        df1['grating_velocity'] = vg
        df1['motor_velocity'] = 0

        df2 = df_decoded_rot_scene[df_decoded_rot_scene['grating_velocity']==vg].copy()

        scene_rotation = df_mean[np.logical_and(df_mean['stim_type']==SCENE_STR, df_mean['vel']==vg)][DECODED_ROTATION].iloc[0] * 180 / np.pi
        true_rotation = 2 * T * vg / np.pi

        sem_scene = df_sem[np.logical_and(df_sem['stim_type']==SCENE_STR, df_sem['vel']==vg)][DECODED_ROTATION].iloc[0] * 180 / np.pi

        fig_path_scene = os.path.join(fig_path_rot_time_series, 'scene_with_actual_HD_shift')
        os.makedirs(fig_path_scene, exist_ok=True)

        fname = 'vg={}.pdf'.format(vg)
        visualization.plot_decoded_and_true_rotation_time_series(df1, df2, color=SCENE_COLOR, true_rotation=true_rotation, decoded_rotation=scene_rotation, sem=sem_scene, fname=fname, figpath=fig_path_scene)


def sensitivity_analysis(loss_fun, model_params, k1=0, k2=5, p1max=1, p2max=1, p1txt=r'$g_{r0}$', p2txt=r'$\gamma_r$', fname='sensitivity_analysis.pdf', fig_path='.'):

    p1 = np.linspace(0, p1max, 21)
    p2 = np.linspace(0, p2max, 21)
    p1v, p2v = np.meshgrid(p1, p2, indexing='ij')
    loss = np.zeros_like(p1v)

    for i in range(len(p1)):
        for j in range(len(p2)):
            p = model_params.copy()
            p[k1] = p1v[i,j]
            p[k2] = p2v[i,j]
            loss[i,j] = loss_fun(p)

    visualization.loss_heatmap(p1, p2, loss, p1fit=model_params[k1], p2fit=model_params[k2], p1txt=p1txt, p2txt=p2txt, fname=fname, fig_path=fig_path)


def evaluate_model_performance(df_bright, df_combined, df_model_est_rotation, df_model_est_rotation_time_series, df_model_eye_vel, sample_weight_vel, sample_weight_dir, out_path='.'):
    rmse_eye_vel_weighted = mean_squared_error(df_combined[EYE_VELOCITY], df_model_eye_vel[EYE_VELOCITY], sample_weight=sample_weight_vel, squared=False)
    rmse_eye_vel = mean_squared_error(df_combined[EYE_VELOCITY], df_model_eye_vel[EYE_VELOCITY], squared=False)
    rmse_rot = mean_squared_error(df_combined[DECODED_ROTATION], df_model_est_rotation_time_series[MODEL_ROTATION], squared=False)
    rmse_rot_weighted = mean_squared_error(df_combined[DECODED_ROTATION], df_model_est_rotation_time_series[MODEL_ROTATION], sample_weight=sample_weight_dir, squared=False)
    rmse_abs_rot = mean_squared_error(df_bright[DECODED_ROTATION], df_model_est_rotation[MODEL_ROTATION], squared=False)
    errors = [180 / np.pi * rmse_rot, 180 / np.pi * rmse_eye_vel, rmse_eye_vel_weighted, 180 / np.pi * rmse_abs_rot, rmse_rot_weighted]
    np.savetxt(os.path.join(out_path, "rmse.csv"), errors, delimiter=",")


def regression_analyses(df, eyetracking=False, frmd7=False, fig_path='.'):

    utils.normalize_restricted(df)
    df = df[df['vel'] > 0]
    df = df[~df['dark']]

    if eyetracking:
        vars = [DECODED_ROTATION, NEG_EYE_ROTATION]
    else:
        vars = [DECODED_ROTATION]

    items = [PLATFORM_STR, SYNC_STR, SCENE_STR]
    txt_locs = [[0.3, 0.25], [0.25, 0.75], [0.4, 0.75]]
    pval_corr = len(items)

    for item, txt_loc in zip(items, txt_locs):

        if isinstance(item, tuple):
            df2 = utils.compute_delta_rotation(df, 'stim_type', [item[0], item[1]])
        else:
            df2 = df[df['stim_type']==item].copy()

        fig_path_reg = os.path.join(fig_path, 'regression_analyses')
        os.makedirs(fig_path_reg, exist_ok=True)

        for var in vars:

            fig_path_reg_var = os.path.join(fig_path_reg, var)
            os.makedirs(fig_path_reg_var, exist_ok=True)

            if isinstance(item, tuple):
                ycol = 'delta_{}'.format(var)
                fname = 'delta_{}_{}.pdf'.format(item[0], item[1])
            else:
                ycol = var
                fname = '{}.pdf'.format(item)

            sm_result = utils.OLS(df2, xcol='vel', ycol=ycol)

            # print(sm_result)
            # print(pval_corr * sm_result.pvalues)

            if frmd7:
                txt_loc = [0.4, 0.25]

            visualization.plot_vel_dependence_with_regression(df2, y=ycol, sm_result=sm_result, reg_color='r', txt_loc=txt_loc, pval_corr=pval_corr, fname=fname, fig_path=fig_path_reg_var)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='combine data from different recordings')
    parser.add_argument('-a', '--animals', nargs='+', required=True)
    parser.add_argument('-t', '--stimset', type=str, required=True)
    parser.add_argument('-e', '--eyetracking', action='store_true')
    parser.add_argument('-d', '--include_dark', action='store_true')
    parser.add_argument('-f', '--frmd7', action='store_true')
    args = parser.parse_args()

    main(args.animals, args.stimset, args.eyetracking, args.model, args.include_dark, args.frmd7)