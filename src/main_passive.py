from vis_vest_hd_computation import analyze_recording, merge_recordings, merge_animals
from vis_vest_hd_computation.imports import *


def main(frmd7=False, include_dark=True, example=False):

    if frmd7:
        data_file = os.path.join(BASE_PATH, 'data', 'FRMD7_passive.csv')
    else:
        data_file = os.path.join(BASE_PATH, 'data', 'WT_passive.csv')

    df = pd.read_csv(data_file, index_col=0)
    df['eye_tracking'] = df['eye_tracking'].astype('bool')
    animals = list(df.index)
    recordings = ['all_combinations_1', 'all_combinations_2', 'all_combinations_3', 'dark_1', 'dark_2']

    rec_list_dark = []

    for a in animals:
        rec_list = []

        for r in recordings:
            recording = df.at[a, r]

            if not pd.isna(recording):

                if r in ['dark_1', 'dark_2']:
                    rec_list_dark.append(recording)
                elif r in ['all_combinations_1', 'all_combinations_2', 'all_combinations_3']:
                    rec_list.append(recording)

                if include_dark:
                    if r in ['dark_1', 'dark_2']:
                        rec_list.append(recording)

                eye_tracking = df.at[a, 'eye_tracking']
                if r in ['dark_1', 'dark_2']:
                    eye_tracking = False
                    dark = True
                else:
                    dark = False

                if not os.path.isdir(os.path.join(BASE_PATH, 'data', 'trial_data', recording)):
                    analyze_recording.main(a, recording, eye_tracking, dark=dark)
                    print(recording)

        eye_tracking = df.at[a, 'eye_tracking']
        merge_recordings.main(a, rec_list, 'all_combinations', eyetracking=eye_tracking, include_dark=include_dark)

    if not example:

        merge_animals.main(animals, 'all_combinations', eyetracking=False, include_dark=include_dark, frmd7=frmd7)

        if not frmd7:
            animals_eye = df.index[df['eye_tracking']]
            merge_animals.main(animals_eye, 'all_combinations', eyetracking=True, include_dark=include_dark, frmd7=frmd7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frmd7', action='store_true')
    parser.add_argument('-d', '--include_dark', action='store_true')
    parser.add_argument('-e', '--example', action='store_true')
    args = parser.parse_args()

    main(frmd7=args.frmd7, include_dark=args.include_dark, example=args.example)
