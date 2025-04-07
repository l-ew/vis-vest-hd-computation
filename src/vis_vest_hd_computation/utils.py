from numpy.polynomial.polynomial import Polynomial
from scipy import stats
from vis_vest_hd_computation.imports import *


def load_dark_data(data_path, data_dir, out_path, animals):
    data_trials_dark = agg_trial_data(data_path, os.path.join(out_path, 'dark'), 'dark', animals, data_dir=data_dir, orient=False, agg_level='animal')
    return data_trials_dark


def load_all_combinations_data(data_path, data_dir, out_path, animals):
    data_trials = agg_trial_data(data_path, os.path.join(out_path, 'all_combinations'), 'all_combinations', animals, data_dir=data_dir, orient=False, agg_level='animal')
    return data_trials


def combine_data(df_decoded, df_eye_vel, df_eye_pos, include_asym=False):
    df_combined = df_eye_vel.copy()
    df_combined[EYE_ROTATION] = df_eye_pos[EYE_ROTATION]
    df_combined[NEG_EYE_ROTATION] = df_eye_pos[NEG_EYE_ROTATION]
    df_combined[DECODED_ROTATION] = df_decoded[DECODED_ROTATION]
    df_combined = df_combined.sort_values(by=['motor_velocity', 'grating_velocity', 'static_grating_interval', 't'])
    return df_combined


def avg_data(df, var=DECODED_ROTATION):
    if 'animal' in df.columns:
        df = avg_data_per_animal(df, var=var)

    cols = ['motor_velocity', 'grating_velocity', 'static_grating_interval', 't']
    cols = list(set(cols).intersection(set(df.columns)))

    df = df.groupby(cols)[var].agg('mean').reset_index()
    return df


def avg_data_per_animal(df, var=DECODED_ROTATION):
    cols = ['animal', 'motor_velocity', 'grating_velocity', 'static_grating_interval', 't']
    cols = list(set(cols).intersection(set(df.columns)))
    df = df.groupby(cols)[var].agg('mean').reset_index()
    return df


def align_and_avg_scene_rotations(df):
    cols = ['animal', 'motor_velocity', 'grating_velocity', 'static_grating_interval', 't']
    cols = list(set(cols).intersection(set(df.columns)))
    mask = np.logical_and(df['motor_velocity'] == 0, df['grating_velocity'] > 0)
    df.loc[mask, 'grating_velocity'] *= -1
    vars = []
    for var in [DECODED_ROTATION, MODEL_ROTATION, NEG_EYE_ROTATION, 'eye_velocity']:
        if var in df.columns:
            df.loc[mask, var] *= -1
            vars.append(var)
    df = df.groupby(cols)[vars].agg('mean').reset_index()
    return df


def trim_data(df, T=T_TRANSIENT_STIM, t_post=0):
    eps = 1e-9
    mask = np.logical_and(df['t'] >= 0, df['t'] <= T + t_post + eps)
    df = df[mask].copy().reset_index()
    return df


def load_data(data_path, items, data_type='agg_data', agg_level='recordings', data_dir=None):
    df_list = []
    for r in items:
        if agg_level=='recordings':
            df_rec = pd.read_pickle(os.path.join(data_path, r, '{}.p'.format(data_type)))
        else:
            df_rec = pd.read_pickle(os.path.join(data_path, r, data_dir, '{}.p'.format(data_type)))
        df_list.append(df_rec)
    df = pd.concat(df_list, ignore_index=True)
    return df


def comp_tuning_curves(x, act, bin_edges=None, n_bins=36):
    """
    Computes tuning curves of neuronal activity as a function of head direction.

    Parameters:
    x: array-like
        Direction data.
    act: array-like
        Neuronal activity data with shape (n_samples, n_cells).
    bin_edges: array-like
        Edges of the bins for head direction.

    Returns:
    tuple
        Contains the following elements:
        - tuning: array
            Normalized tuning curves for each cell.
        - bin_centers: array
            Centers of the head direction bins.
    """

    if bin_edges is None: 
        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)

    n_cells = act.shape[1]
    activities = [act[:,i].squeeze() for i in range(n_cells)]
    tuning, _, _ = stats.binned_statistic(x, activities, statistic='mean', bins=bin_edges)
    bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    norm_const = np.max(tuning, axis=1)
    tuning = tuning / norm_const[:, np.newaxis]
    return tuning, bin_centers


def normalize_tuning(tuning):
    norm_const = np.max(tuning, axis=1)
    return tuning / norm_const[:, np.newaxis]


def load_stimuli(experiment_path, stim_orientation):
    """
    Loads stimuli data for an experiment.

    Parameters:
    experiment_path: str
        Path to the experiment directory containing the stimuli data files.
    stim_orientation: int
        Orientation multiplier for the stimulus directions.

    Returns:
    tuple
        Contains the following elements:
        - motor_vels: array
            Array of motor velocities.
        - motor_directions: array
            Array of processed motor directions.
        - grating_vels: array
            Array of grating velocities.
        - grating_directions: array
            Array of processed grating directions.
    """

    motor_vels = np.load(os.path.join(experiment_path, 'motor_speeds.npy'))
    grating_vels = np.load(os.path.join(experiment_path, 'grating_speeds.npy'))

    with np.load(os.path.join(experiment_path, 'motor_directions.npz')) as npz:
        motor_directions = np.ma.MaskedArray(**npz)

    with np.load(os.path.join(experiment_path, 'grating_directions.npz')) as npz:
        grating_directions = np.ma.MaskedArray(**npz)

    motor_directions.data[motor_directions.mask] = 0
    motor_directions = stim_orientation * motor_directions.data

    grating_directions.data[grating_directions.mask] = 0
    grating_directions = stim_orientation * grating_directions.data

    return motor_vels, motor_directions, grating_vels, grating_directions


def align_restricted(df, inplace=True):
    """
    Aligns the data in a DataFrame from clockwise and counterclockwise stimulus rotations.

    Parameters:
    df: pandas.DataFrame
        DataFrame containing the data to be aligned.
    inplace: bool, optional
        Flag if the DataFrame should be manipulated inplace.

    Returns:
    pandas.DataFrame
        The modified DataFrame with aligned data.
    """

    if not inplace:
        df = df.copy()

    mask = df['vel'] < 0
    df.loc[mask, 'vel'] *= -1
    scene_mask = df['stim_type'] == SCENE_STR

    for var in [DECODED_ROTATION, MODEL_ROTATION, NEG_EYE_ROTATION, MODEL_NEG_EYE_ROTATION, 'rotation']:
        if var in df.columns:
            df.loc[mask, var] *= -1
            df.loc[scene_mask, var] *= -1

    return df


def normalize_restricted(df, T=T_TRANSIENT_STIM):
    """
    Normalizes the specified variable in the DataFrame based on the total stimulus rotation.

    Parameters:
    df: pandas.DataFrame
        DataFrame containing the data to be normalized.
    T: float, optional
        Stimulus duration. Default is T_TRANSIENT_STIM.
    var: str, optional
        Name of the column to be normalized. Default is DECODED_ROTATION.

    Returns:
    pandas.DataFrame
        The modified DataFrame with normalized data.
    """

    ref_rotation = df['vel'] * T / 90
    for var in [DECODED_ROTATION, NEG_EYE_ROTATION, MODEL_ROTATION, MODEL_NEG_EYE_ROTATION, 'rotation']:
        if var in df.columns:
            df[var] = 100 * df[var] / ref_rotation


def comp_peak_firing_direction(tuning, hd_bin_centers):
    max_ind = np.argmax(tuning, axis=1)
    ordering = np.argsort(max_ind)
    peak_firing_dir = [hd_bin_centers[k] for k in max_ind]
    return np.array(peak_firing_dir), ordering


def data_string(data_dir, eyetracking, frmd7, include_asymmetric=False):
    if eyetracking:
        eyestr = '_eye'
    else:
        eyestr = ''

    if frmd7:
        group = '_FRMD7'
    else:
        group = '_WT'

    if include_asymmetric:
        asym_str = '_asym'
    else:
        asym_str = ''

    return data_dir + eyestr + group + asym_str


def get_fig_path(base_path, data_dir, eyetracking, frmd7, include_asymmetric=False):
    str = data_string(data_dir, eyetracking, frmd7, include_asymmetric=include_asymmetric)
    fig_path = os.path.join(base_path, 'figures', str)
    os.makedirs(fig_path, exist_ok=True)
    return fig_path


def get_out_path(base_path, data_dir, eyetracking, frmd7, include_asymmetric=False):
    str = data_string(data_dir, eyetracking, frmd7, include_asymmetric=include_asymmetric)
    out_path = os.path.join(base_path, 'data', 'trial_data', str)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def comp_prop_factor(df, xcol, ycol):
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    from sklearn.linear_model import HuberRegressor
    x = x[:,np.newaxis]
    huber = HuberRegressor(epsilon=1.35, fit_intercept=False).fit(x, y)
    return huber.coef_[0]


def comp_eye_rotation(coef, coef2, T=T_TRANSIENT_STIM):
    """
    Computes the total eye rotation based on polynomial coefficients and the stimulus duration.

    Parameters:
    coef: array-like
        Coefficients of the polynomial representing the first part of the eye rotation.
    coef2: array-like
        Coefficients of the polynomial representing the second part of the eye rotation.
    T: float, optional
        Stimulus duration. Default is T_TRANSIENT_STIM.

    Returns:
    float
        The computed total eye rotation, or NaN if the input coefficients are invalid.
    """

    if np.any(np.isnan(coef)) or len(coef) == 0:
        return np.nan
    elif len(coef2) == 0:
        degree = len(coef) - 1
        k = np.arange(degree + 1)
        return np.sum(coef * T**(k+1) / (k + 1))
    else:
        degree = len(coef) - 1
        k = np.arange(degree + 1)
        a = np.sum(coef * (0.5*T)**(k+1) / (k + 1))
        b = np.sum(coef2 * T**(k+1) / (k + 1)) - np.sum(coef2 * (0.5*T)**(k+1) / (k + 1))
        return a + b


def fit_fun(t, amp, phase, T=T_TRANSIENT_STIM):
    y = amp * np.sin(np.pi * t / T + phase)
    y[np.sign(y) != np.sign(amp)] = 0
    return y


def get_data_str(stimset, include_dark):
    if include_dark:
        darkstr = '_dark'
    else:
        darkstr = ''

    out_dir = stimset + darkstr
    return out_dir


def get_vmax(stimset):
    if stimset == 'constant':
        v_max = [0, 18]
    else:
        v_max = [0, 45, 90, 135, 180]
    return v_max


def get_vel_combs(stimset, v_max):
    vels = [-x for x in v_max[::-1] if x != 0] + v_max
    vm, vg = np.meshgrid(vels, vels)
    vm, vg = vm.flatten(), vg.flatten()

    if stimset == 'restricted':
        mask = np.logical_or.reduce((vm == 0, vg == 0, vm == vg))
        vm, vg = vm[mask], vg[mask]

    return vm, vg


def save_trial_data(data_path, data, v_max, asymmetric=False):
    for data_stim in data:
        vm = v_max[data_stim['motor_velocity']]
        vg = v_max[data_stim['grating_velocity']]
        dm = data_stim['motor_direction']
        dg = data_stim['grating_direction']
        sgi = data_stim['static_grating_interval']

        if asymmetric:
            fname = 'vm={}_vg={}_static_grating_interval={}.p'.format(dm * vm, dg * vg, sgi)
        else:
            fname = 'vm={}_vg={}.p'.format(dm * vm, dg * vg)
        with open(os.path.join(data_path, fname), 'wb') as f:
            pickle.dump(data_stim, f)


def orient_data(data, data_path):
    orientation_file = os.path.join(data_path, 'orientation.txt')
    with open(orientation_file) as f:
        orientation = int(f.readlines()[0])

    data['before'] = [orientation * x for x in data['before']]
    data['after'] = [orientation * x for x in data['after']]
    data[DECODED_DIRECTION] = [orientation * x for x in data[DECODED_DIRECTION]]
    return data


def file_path_helper(data_path, group, data_dir, stimset, fname, agg_level='recording'):
    if agg_level == 'animal':
        file_path = os.path.join(data_path, group, data_dir, stimset, fname)
    else:
        file_path = os.path.join(data_path, group, data_dir, fname)
    return file_path


def agg_trial_data(data_path, out_path, stimset, groups, data_dir='', orient=True, agg_level='recording', asymmetric=False):

    v_max = get_vmax(stimset=stimset)
    vms, vgs = get_vel_combs(stimset, v_max)

    os.makedirs(out_path, exist_ok=True)

    data = []

    for vm, vg in zip(vms, vgs):
        if asymmetric:
            for static_grating_interval in [0, 1]:
                fname = 'vm={}_vg={}_static_grating_interval={}.p'.format(vm, vg, static_grating_interval)
                file_path = file_path_helper(data_path, groups[0], data_dir, stimset, fname, agg_level=agg_level)
                if (vm != 0 or vg == 0) and os.path.isfile(file_path):
                    data_stim = load_stim_data(data_path, file_path, stimset, groups, data_dir, fname, orient=orient, agg_level=agg_level)
                    save_trial_data(out_path, [data_stim], v_max)
                    data.append(data_stim)
        else:
            fname = 'vm={}_vg={}.p'.format(vm, vg)
            file_path = file_path_helper(data_path, groups[0], data_dir, stimset, fname, agg_level=agg_level)
            if os.path.isfile(file_path):
                data_stim = load_stim_data(data_path, file_path, stimset, groups, data_dir, fname, orient=orient, agg_level=agg_level)
                save_trial_data(out_path, [data_stim], v_max)
                data.append(data_stim)

    return data


def load_stim_data(data_path, file_path, stimset, groups, data_dir, fname, agg_level='recording', orient=True):

    with open(file_path, 'rb') as f:
        data_stim = pickle.load(f)

    data_stim[agg_level] = [groups[0]] * len(data_stim[DECODED_DIRECTION])

    common_keys = set(data_stim.keys())

    if orient:
        data_stim = orient_data(data_stim, os.path.join(data_path, groups[0], data_dir))

    for group in groups[1:]:

        file_path = file_path_helper(data_path, group, data_dir, stimset, fname, agg_level=agg_level)

        with open(file_path, 'rb') as f:
            add_data = pickle.load(f)

        add_data[agg_level] = [group] * len(add_data[DECODED_DIRECTION])

        if orient:
            add_data = orient_data(add_data, os.path.join(data_path, group, data_dir))

        common_keys.intersection_update(set(add_data.keys()))

        for key in common_keys:

            if isinstance(data_stim[key], list) or isinstance(data_stim[key], np.ndarray):
                data_stim[key] += add_data[key]

    data_stim = dict((k, data_stim[k]) for k in common_keys if k in data_stim)

    return data_stim


def get_rotation_time_series(data_trials, T=T_TRANSIENT_STIM, v_max=None, agg_dirs=True, fps=FPS_MINISCOPE, offset=1, dark=False):
    """
    Generates a time series DataFrame for rotation data from multiple trials.

    Parameters:
    data_trials: list of dict
        List of dictionaries containing trial data. Each dictionary should have keys 'motor_direction', 'grating_direction',
        'motor_velocity', 'grating_velocity', 'static_grating_interval', 'animal', 'recording', 't', and DECODED_DIRECTION.
    T: float, optional
        Transient stimulus time. Default is T_TRANSIENT_STIM.
    agg_dirs: bool, optional
        If True, aggregate directions by flipping negative motor velocities. Default is True.
    fps: int, optional
        Frames per second. Default is FPS_MINISCOPE.
    offset: int, optional
        Offset time before and after the stimulus. Default is 1.
    dark: bool, optional
        Indicates whether the trials were performed in dark conditions. Default is False.

    Returns:
    pandas.DataFrame
        A DataFrame containing the time series data for the decoded rotation with columns for animal, recording, motor velocity,
        grating velocity, static grating interval, and decoded rotation.
    """

    n_steps = int(fps * (T + 2 * offset) + 1)
    t = np.linspace(-offset, T+offset, n_steps)

    animal_key = 'animal' in data_trials[0].keys()

    motor_velocity = []
    grating_velocity = []
    internal_dir = []
    static_grating_interval = []
    recording = []
    trial_id = []
    if animal_key:
        animal = []

    if v_max is None:
        v_max = [0, 45, 90, 135, 180]
    for data_stim in data_trials:

        dm = data_stim['motor_direction']
        dg = data_stim['grating_direction']
        vm = dm * v_max[data_stim['motor_velocity']]
        vg = dg * v_max[data_stim['grating_velocity']]
        sgi = data_stim['static_grating_interval']
        r = data_stim['recording']

        if animal_key:
            a = data_stim['animal']
            animal += a

        recording += r
        trial_id += data_stim['trial_id']

        for t_trial, internal_dir_trial, before in zip(data_stim['t'], data_stim[DECODED_DIRECTION], data_stim['before']):

            if agg_dirs and vm < 0:
                internal_dir_trial = -internal_dir_trial
                before = -before
                motor_velocity.append(-vm)
                grating_velocity.append(-vg)
            else:
                motor_velocity.append(vm)
                grating_velocity.append(vg)

            internal_dir_trial = internal_dir_trial - before
            internal_dir_trial = np.interp(t, t_trial, internal_dir_trial)
            internal_dir.append(internal_dir_trial)
            static_grating_interval.append(sgi)
         
    df = pd.DataFrame()
    if animal_key:
        df['animal'] = animal
    df['recording'] = recording
    df['trial_id'] = trial_id
    df['motor_velocity'] = motor_velocity
    df['grating_velocity'] = grating_velocity
    df['static_grating_interval'] = static_grating_interval

    internal_dir = np.vstack(internal_dir)

    t_columns = ['t_{}'.format(k) for k in range(n_steps)]
    df_temp = pd.DataFrame(internal_dir, columns=t_columns)
    if animal_key:
        colnames = ['animal', 'recording', 'trial_id', 'motor_velocity', 'grating_velocity', 'static_grating_interval']
    else:
        colnames = ['recording', 'trial_id', 'motor_velocity', 'grating_velocity', 'static_grating_interval']
    df_temp = pd.concat([df, df_temp], axis=1)
    df_temp = pd.melt(df_temp, id_vars=colnames, value_vars=t_columns, var_name='t', value_name=DECODED_ROTATION)
    df_temp['t'] = df_temp['t'].apply(lambda x: (T + 2 * offset) * float(x[2:]) / (n_steps - 1) - offset)
    #df_temp['animal'] = 'n/a'
    df_temp['dark'] = dark

    return df_temp


def get_temp_df(df, trace, T, n_steps, dark=False, value_name='eye_velocity'):
    t_columns = ['t_{}'.format(k) for k in range(n_steps)]
    df_temp = pd.DataFrame(trace, columns=t_columns)

    if 'static_grating_interval' in df.columns:
        colnames = ['motor_velocity', 'grating_velocity', 'static_grating_interval']
    else:
        colnames = ['motor_velocity', 'grating_velocity']
    df_temp = pd.concat([df[colnames].reset_index(drop=True), df_temp], axis=1)

    df_temp = pd.melt(df_temp, id_vars=colnames, value_vars=t_columns, var_name='t', value_name=value_name)
    df_temp['t'] = df_temp['t'].apply(lambda x: T * float(x[2:]) / (n_steps - 1))
    df_temp['dark'] = dark

    df_temp = df_temp.sort_values(by=colnames)

    return df_temp


def align_traces(internal_dir, ref_dir):
    y = internal_dir.copy() - ref_dir
    return np.angle(np.exp(1j * y))


def direction_str(x):
    if x == 1:
        return 'cw'
    elif x == -1:
        return 'ccw'
    else:
        return ''


def error_fun(x):
    x_median = np.median(x)
    mad = np.median(np.abs(x - x_median))
    xmin = x_median - mad
    xmax = x_median + mad
    return (xmin, xmax)


def extract_restricted_data(df, dark=False):
    """
    Extracts stimulus conditions in which either only platform rotates, only the visual scene rotates or both rotate synchroneously.

    Parameters:
    df: pandas.DataFrame
        DataFrame containing the data to be filtered and categorized.
    dark: bool, optional
        If True, only includes 'motor' stim_type in the duplicated 'catch' trials. Default is False.

    Returns:
    pandas.DataFrame
        The DataFrame containing only data from the three specified stimulus conditions.
    """

    mask1 = np.array(df['grating_velocity']) == 0
    mask2 = np.array(df['grating_velocity']) == np.array(df['motor_velocity'])
    mask3 = np.array(df['motor_velocity']) == 0
    mask = np.logical_or.reduce((mask1, mask2, mask3))
    df_restricted = df.copy()

    vel = np.zeros(len(mask), dtype='int')
    vel[mask1] = df['motor_velocity'][mask1]
    vel[mask2] = df['motor_velocity'][mask2]
    vel[mask3] = df['grating_velocity'][mask3]
    df_restricted['vel'] = vel

    df_restricted['stim_type'] = ''
    df_restricted.loc[mask1, 'stim_type'] = PLATFORM_STR
    df_restricted.loc[mask2, 'stim_type'] = SYNC_STR
    df_restricted.loc[mask3, 'stim_type'] = SCENE_STR

    mask_catch = np.logical_and(mask2, mask3)
    df_restricted.loc[mask_catch, 'stim_type'] = 'catch'
    df_restricted = df_restricted[mask]

    mask_catch = df_restricted['stim_type'] == 'catch'
    df_tmp = df_restricted[mask_catch].copy()
    df_tmp['stim_type'] = PLATFORM_STR
    df_restricted = pd.concat([df_restricted, df_tmp], ignore_index=True)
    if not dark:
        df_tmp['stim_type'] = SCENE_STR
        df_restricted = df = pd.concat([df_restricted, df_tmp], ignore_index=True)
        df_tmp['stim_type'] = SYNC_STR
        df_restricted = pd.concat([df_restricted, df_tmp], ignore_index=True)

    mask = df_restricted['stim_type'] != 'catch'
    df_restricted = df_restricted[mask]

    return df_restricted


def get_size(pvalue_str):
    if pvalue_str == 'ns':
        return 7
    else:
        return 9


def convert_pvalue_to_asterisks(pvalue):
    if pvalue < 1e-6:
        return "6*"
    elif pvalue < 1e-5:
        return "5*"
    elif pvalue < 0.0001:
        return "****"
    elif pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.05:
        return "*"
    return "ns"


def convert_pvalue_str(pvalue):
    if pvalue <= 0.0001:
        return r'$p < 0.0001$'
    elif pvalue <= 0.001:
        return r'$p < 0.001$'
    elif pvalue <= 0.01:
        return r'$p < 0.01$'
    elif pvalue <= 0.05:
        return r'$p < 0.05$'
    return "ns"


def comp_eye_time_series(df, T=T_TRANSIENT_STIM, fps=FPS_MINISCOPE, asymmetric=False, aggregate=True):
    mask = np.logical_not(np.isnan(df['quick_phase_count']))
    df = df[mask].reset_index()
    if aggregate:
        df = align_rotations(df, 'motor_velocity')

    colnames = ['animal', 'motor_velocity', 'grating_velocity', 'static_grating_interval', 'eye_velocity_coef', 'eye_velocity_coef_2nd_half']
    df_eyevel = df[colnames].copy()
    df_eyepos = df[colnames].copy()

    n_pts = int(T * fps + 1)
    t_interp = np.linspace(0, T, n_pts)
    dt = 1 / fps

    n_rows, n_cols = df_eyevel.shape
    t_columns = ['t_{}'.format(k) for k in range(n_pts)]
    df_eyevel = pd.concat([df_eyevel, pd.DataFrame(columns=t_columns)], axis=1)
    df_eyepos = pd.concat([df_eyepos, pd.DataFrame(columns=t_columns)], axis=1)
    t_col_ids = np.arange(n_cols, n_cols + n_pts)

    for k in range(n_rows):
        coef = df_eyevel['eye_velocity_coef'].iloc[k]
        p = Polynomial(coef)
        eye_speed = p(t_interp)

        coef2 = df_eyevel['eye_velocity_coef_2nd_half'].iloc[k]

        if len(coef2) > 0:
            p = Polynomial(coef2)
            mask = t_interp > 0.5 * T
            eye_speed[mask] = p(t_interp[mask])

        df_eyevel.iloc[k, t_col_ids] = eye_speed
        df_eyepos.iloc[k, t_col_ids] = np.cumsum(eye_speed * dt, axis=-1)

    df_eyevel = pd.melt(df_eyevel, id_vars=colnames, value_vars=t_columns, var_name='t', value_name=EYE_VELOCITY)
    df_eyevel['t'] = df_eyevel['t'].apply(lambda x: T * float(x[2:]) / (n_pts - 1))

    df_eyepos = pd.melt(df_eyepos, id_vars=colnames, value_vars=t_columns, var_name='t', value_name=EYE_ROTATION)
    df_eyepos['t'] = df_eyepos['t'].apply(lambda x: T * float(x[2:]) / (n_pts - 1))
    df_eyepos[NEG_EYE_ROTATION] = -df_eyepos[EYE_ROTATION] 

    df_eyevel['dark'] = False
    df_eyepos['dark'] = False

    return df_eyevel, df_eyepos


def _get_head_trajectory_sin(v, T, n=3):
    t = np.linspace(0,T,1000)
    x = -v * T * (np.cos(np.pi * t / T) - 1) / 180
    t = np.hstack((-n,t,T+n))
    x = np.hstack((0,x,x[-1]))
    return t, x


def get_head_trajectory(v, T, n=1, const=False):
    return _get_head_trajectory_sin(v, T, n=n)


def align_rotations(df, alignment_var, inplace=True):
    """
    Aligns the data in the DataFrame from clockwise and counterclockwise stimulus rotations.

    Parameters:
    df: pandas.DataFrame
        DataFrame containing the data to be aligned.
    alignment_var: str
        Name of the column used for alignment.
    inplace: bool, optional
        If False, creates a copy of the DataFrame before aligning directions. Default is True.

    Returns:
    pandas.DataFrame
        The modified DataFrame with aligned data.
    """

    if not inplace:
        df = df.copy()

    mask = df[alignment_var] < 0
    df.loc[mask, 'motor_velocity'] *= -1
    df.loc[mask, 'grating_velocity'] *= -1

    for col in [DECODED_ROTATION, MODEL_ROTATION, NEG_EYE_ROTATION, 'median_eye_velocity', 'eye_turn_1st_half']:
        if col in df.columns:
            df.loc[mask, col] *= -1

    if 'eye_velocity_coef' in df.columns:
        df.loc[mask, 'eye_velocity_coef'] = df.loc[mask, 'eye_velocity_coef'].apply(lambda x: (-np.array(x)).tolist())
        df.loc[mask, 'eye_velocity_coef_2nd_half'] = df.loc[mask, 'eye_velocity_coef_2nd_half'].apply(lambda x: (-np.array(x)).tolist())

    return df


def normalize_rotations(df, var, T=T_TRANSIENT_STIM, inplace=True, invert=False, drop_invalid=False):
    """
    Normalizes rotations in the DataFrame based on a specified variable.

    Parameters:
    df: pandas.DataFrame
        DataFrame containing the data to be normalized.
    var: str
        Name of the column used to determine the reference rotation.
    T: float, optional
        Transient stimulus time. Default is T_TRANSIENT_STIM.
    inplace: bool, optional
        If True, modifies the DataFrame in place. Otherwise, returns a modified copy. Default is True.
    invert: bool, optional
        If True, inverts the reference rotation. Default is False.
    drop_invalid: bool, optional
        If True, drops rows where the specified variable is zero. Otherwise, sets them to NaN. Default is False.

    Returns:
    pandas.DataFrame
        The DataFrame with normalized rotations.
    """

    if not inplace:
        df2 = df.copy()
    else:
        df2 = df

    mask = df[var] == 0
    if drop_invalid:
        df2.drop(df2[mask].index, inplace=True)
    else:
        df2.loc[mask, var] = np.nan

    mask = df2[var] != 0
    ref_rotation = (1 - 2 * invert) * df2.loc[mask, var] * T / 90

    for col in [DECODED_ROTATION, MODEL_ROTATION, NEG_EYE_ROTATION, MODEL_NEG_EYE_ROTATION]:
        if col in df2.columns:
            df2.loc[mask, col] *= 100 / ref_rotation

    return df2


def circ_corr(x, y):
    """
    Compute the circular correlation coefficient between two circular variables.

    Parameters:
    x : array-like
        The first circular variable (angles in radians).
    y : array-like
        The second circular variable (angles in radians).

    Returns:
    float
        The circular correlation coefficient between the two variables.
    """

    x = x[:, np.newaxis] - x[np.newaxis, :]
    y = y[:, np.newaxis] - y[np.newaxis, :]
    x = np.sin(x.ravel())
    y = np.sin(y.ravel())
    corr = np.nansum(x * y) / (np.sqrt(np.nansum(x**2)) * np.sqrt(np.nansum(y**2)))
    return corr


def circ_corr_mem(x, y):
    """
    Compute the circular correlation coefficient between two circular variables. Use less memory.

    Parameters:
    x : array-like
        The first circular variable (angles in radians).
    y : array-like
        The second circular variable (angles in radians).

    Returns:
    float
        The circular correlation coefficient between the two variables.
    """

    interaction_mean = 0.
    x_mean = 0.
    y_mean = 0.
    n = len(x)
    for k, (xi, yi) in enumerate(zip(x,y)):
        dx = x - xi
        dy = y - yi
        sin_dx = np.sin(dx)
        sin_dy = np.sin(dy)
        fac = 1 / ((k + 1) * n)
        interaction_mean += fac * (np.sum(sin_dx * sin_dy) - n * interaction_mean)
        x_mean += fac * (np.sum(sin_dx**2) - n * x_mean)
        y_mean += fac * (np.sum(sin_dy**2) - n * y_mean)

    corr = interaction_mean / (np.sqrt(x_mean) * np.sqrt(y_mean))

    return corr


def get_true_rotation_time_series(T=3):

    ts = []
    rotation = []
    v = []
    for vm in [0, 45, 90, 135, 180]:
        t, x = _get_head_trajectory_sin(vm, T, n=1)
        ts.append(t)
        rotation.append(x)
        v.append([vm] * len(t))

    ts = np.hstack(ts)
    rotation = np.hstack(rotation)
    v = np.hstack(v)

    df_true_rotation = pd.DataFrame()
    df_true_rotation['animal'] = 'n/a'
    df_true_rotation['t'] = ts
    df_true_rotation[TRUE_ROTATION] = rotation
    df_true_rotation['motor_velocity'] = v
    df_true_rotation['grating_velocity'] = 0
    df_true_rotation['static_grating_interval'] = -1

    return df_true_rotation


def rayleigh_vector(activity, theta):
    """
    Compute Rayleigh vectors from binned activity values.

    Parameters:
    activity : numpy.ndarray
        The binned activity values. Shape should be (n_cells, n_bins).
    theta : numpy.ndarray
        The angles corresponding to each bin. Shape should be (n_bins,).

    Returns:
    tuple
        pref_dir : numpy.ndarray
            The preferred direction for each sample. Shape is (n_cells,).
        vec_length : numpy.ndarray
            The vector length for each sample. Shape is (n_cells,).
    """

    z = np.nansum(activity * np.exp(1j * theta[np.newaxis, :]), axis=1)
    z = z / np.nansum(activity, axis=1)
    pref_dir = np.angle(z, deg=False)
    vec_length = np.abs(z)
    return pref_dir, vec_length


def comp_delta(x, y):
    """
    Compute the circular difference between two angles, ensuring the result 
    is within the range [-π, π].

    Parameters:
    x : float or array-like
        The first angle(s) in radians.
    y : float or array-like
        The second angle(s) in radians.

    Returns:
    float or array-like
        The circular difference between the angles, wrapped to the range [-π, π].
    """
    delta = x - y
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    return delta


def map_angle(x):
    x += 2 * np.pi
    x %= 2 * np.pi
    return x


def compute_delta_rotation(df, col, vals):
    df0 = df[df[col] == vals[0]].copy()
    df1 = df[df[col] == vals[1]].copy()

    if col != 'vel':
        ind_cols = ['animal', 'vel']
    else:
        ind_cols = ['animal']

    df0 = df0.set_index(ind_cols)
    df1 = df1.set_index(ind_cols)
    df_joined = df0.join(df1, lsuffix='_{}'.format(vals[0]), rsuffix='_{}'.format(vals[1]))

    decoded0 = '{}_{}'.format(DECODED_ROTATION, vals[0])
    decoded1 = '{}_{}'.format(DECODED_ROTATION, vals[1])
    df_joined['delta_{}'.format(DECODED_ROTATION)] = df_joined[decoded0] - df_joined[decoded1]

    if NEG_EYE_ROTATION in df.columns:
        eye0 = '{}_{}'.format(NEG_EYE_ROTATION, vals[0])
        eye1 = '{}_{}'.format(NEG_EYE_ROTATION, vals[1])
        df_joined['delta_{}'.format(NEG_EYE_ROTATION)] = df_joined[eye0] - df_joined[eye1]

    return df_joined.reset_index()


def OLS(df, xcol='vel', ycol=DECODED_ROTATION):

    import statsmodels.api as sm
    X = df[xcol]
    Y = df[ycol]
    X = sm.add_constant(X)

    linear_model = sm.OLS(Y, X).fit()
    print(linear_model.summary())
    return linear_model
