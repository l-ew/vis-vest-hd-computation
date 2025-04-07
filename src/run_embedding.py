from vis_vest_hd_computation.imports import *
import vis_vest_hd_computation.spectral_embedding as spectral_embedding


def main():

    datasets = ['WT_active', 'WT_passive', 'FRMD7_active', 'FRMD7_passive']
    path = os.path.join(BASE_PATH, 'data', 'base_data')

    for dataset in datasets:
        if 'active' in dataset:
            recordings = ['active_3landmarks', 'combined', 'active_grating_stim']
        else:
            recordings = ['all_combinations_1', 'all_combinations_2', 'all_combinations_3', 'dark_1', 'dark_2']

        data_file = os.path.join(BASE_PATH, 'data', '{}.csv'.format(dataset))
        df = pd.read_csv(data_file, index_col=0)
        animals = list(df.index)

        for a in animals:
            for r in recordings:
                recording = df.at[a, r]
                if not pd.isna(recording):

                    print(recording)

                    opt = False  # Note: setting this to "True" will iterate over a grid of possible values for the global scale parameter "eps" (see Methods)

                    if r == 'active_3landmarks' or r == 'combined':
                        orient = True
                        trim = False
                        align = True
                    elif r == 'active_grating_stim':
                        orient = True
                        trim = False
                        align = False
                    else:
                        orient = False
                        trim = True
                        align = False

                    spectral_embedding.main(recording, path=path, opt=opt, eps=0.03, smooth=True, orient=orient, trim=trim, align=align)


if __name__ == "__main__":
    main()
