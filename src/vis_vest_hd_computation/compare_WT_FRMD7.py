import vis_vest_hd_computation.utils as utils
import matplotlib.markers as mmarkers
from vis_vest_hd_computation.imports import *
import vis_vest_hd_computation.visualization as visualization

def main():
    stimset = 'all_combinations'
    include_dark = True
    data_dir = utils.get_data_str(stimset, include_dark)
    data_path = os.path.join(BASE_PATH, 'data', 'trial_data')
    
    fig_path = os.path.join(BASE_PATH, 'figures', 'WT_vs_FRMD7')
    os.makedirs(fig_path, exist_ok=True)

    file_path = os.path.join(data_path, data_dir + '_WT', 'restricted_stimset.p')
    with open(file_path, 'rb') as f:
        df_WT = pickle.load(f)
    df_WT['phenotype'] = 'WT'

    file_path = os.path.join(data_path, data_dir + '_FRMD7', 'restricted_stimset.p')
    with open(file_path, 'rb') as f:
        df_FRMD7 = pickle.load(f)
    df_FRMD7['phenotype'] = 'FRMD7'

    df = pd.concat([df_WT, df_FRMD7])
    df = utils.align_restricted(df)
    df = df.groupby(['phenotype', 'animal', 'stim_type', 'vel', 'dark'])[DECODED_ROTATION].agg('mean').reset_index()
    df = df[df['vel']!=0]
    utils.normalize_restricted(df)

    style = 'phenotype'
    style_order = ['WT', 'FRMD7']
    markers = {'WT': mmarkers.MarkerStyle('v').scaled(0.75), 'FRMD7': mmarkers.MarkerStyle('^').scaled(0.75)}

    df_dark = df[df['dark']].copy()

    visualization.dark_rotation_comparison(df_dark, style, style_order, markers, fname='WT_FRMD7_{}.pdf'.format(DECODED_ROTATION), fig_path=fig_path)

    compare_rotation_over_time_dark(data_path, data_dir, fig_path=fig_path)


def compare_rotation_over_time_dark(data_path, data_dir, fig_path='.'):

    file_path = os.path.join(data_path, data_dir + '_WT', 'decoded_rot_time_series_dark.p')
    with open(file_path, 'rb') as f:
        df_WT = pickle.load(f)
    df_WT['phenotype'] = 'WT'

    file_path = os.path.join(data_path, data_dir + '_FRMD7', 'decoded_rot_time_series_dark.p')
    with open(file_path, 'rb') as f:
        df_FRMD7 = pickle.load(f)
    df_FRMD7['phenotype'] = 'FRMD7'

    df_dark = pd.concat([df_WT, df_FRMD7])

    for vm in [45, 90 ,135, 180]:
        fname = '{}_time_series_dark_vm={}.pdf'.format(DECODED_ROTATION.capitalize().replace('_', ' '), vm)
        visualization.plot_rotation_over_time_dark(df_dark[df_dark['motor_velocity']==vm].copy(), color=DARK_COLOR, fname=fname, figpath=fig_path)


if __name__ == "__main__":
    main()
