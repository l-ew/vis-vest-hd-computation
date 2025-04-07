import vis_vest_hd_computation.visualization as visualization
import vis_vest_hd_computation.utils as utils
from vis_vest_hd_computation.imports import *


def main(animal, recordings, stimset, eyetracking=False, include_dark=True):

    out_dir = utils.get_data_str(stimset, include_dark)

    data_path = os.path.join(BASE_PATH, 'data', 'trial_data')
    fig_path = os.path.join(BASE_PATH, 'figures', animal, out_dir)
    os.makedirs(fig_path, exist_ok=True)

    out_path = os.path.join(data_path, animal, out_dir)
    os.makedirs(out_path, exist_ok=True)

    if include_dark:
        dark_recordings = list(filter(lambda x: 'dark' in x, recordings))
        utils.agg_trial_data(data_path, os.path.join(out_path, 'dark'), 'dark', dark_recordings)

    grating_recordings = list(filter(lambda x: 'dark' not in x, recordings))
    utils.agg_trial_data(data_path, os.path.join(out_path, 'all_combinations'), 'all_combinations', grating_recordings)

    df = utils.load_data(data_path, recordings, data_type='restricted_stimset')
    visualization.plot_restricted(df, var=DECODED_ROTATION, fig_path=fig_path)
    if eyetracking:
        visualization.plot_restricted(df, var=NEG_EYE_ROTATION, fig_path=fig_path)

    df_fname = 'restricted_stimset.p'
    df.to_pickle(os.path.join(out_path, df_fname))

    df_all_comb = utils.load_data(data_path, recordings, data_type='agg_data')

    df_fname = 'agg_data.p'
    df_all_comb.to_pickle(os.path.join(out_path, df_fname))

    df_all_comb = df_all_comb[~df_all_comb['dark']]
    visualization.plot_all_combinations(df_all_comb, var=DECODED_ROTATION, fig_path=fig_path)
    if eyetracking:
        visualization.plot_all_combinations(df_all_comb, var=NEG_EYE_ROTATION, fig_path=fig_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge data from different recordings')
    parser.add_argument('-a', '--animal', type=str, required=True)
    parser.add_argument('-s', '--recordings', nargs='+', required=True)
    parser.add_argument('-t', '--stimset', type=str, required=True)
    parser.add_argument('-e', '--eyetracking', action='store_true')
    parser.add_argument('-d', '--include_dark', action='store_true')

    args = parser.parse_args()

    main(args.animal, args.recordings, args.stimset, args.eyetracking, args.include_dark)

