from vis_vest_hd_computation.imports import *
from vis_vest_hd_computation.utils import load_stimuli, comp_eye_rotation, get_head_trajectory, get_vmax, save_trial_data, extract_restricted_data
import vis_vest_hd_computation.visualization as visualization
from scipy.ndimage import label, gaussian_filter1d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import HuberRegressor
import itertools


def extract_trials(t, x, start_inds, stop_inds, stim_list, stim, n=2, fps=12):
    """
    Extracts direction over time and related information for trials of a given stimulus.
    
    Parameters:
    t: array
        Time array.
    x: array
        Direction array.
    start_inds: list
        List of start indices for trials.
    stop_inds: list
        List of stop indices for trials.
    stim_list: tuple
        Tuple containing motor velocities, grating velocities, motor directions,
        grating directions, and static grating intervals.
    stim: tuple
        Tuple containing specific motor velocity, grating velocity, motor direction,
        grating direction, and static grating interval.
    n: int
        Number of frames before and after to include in the trials. Default is 2.
    fps: int
        Frames per second. Default is 12.
    
    Returns:
    dict: Dictionary containing extracted trial data and related information.
    """

    motor_vels, grating_vels, motor_directions, grating_directions, static_grating_intervals = stim_list
    motor_vel, grating_vel, motor_direction, grating_direction, static_grating_interval = stim

    direction_trials = []
    t_trials = []
    before = []
    after = []
    trial_id = []

    for i, (vm, vg, dm, dg, sgi) in enumerate(zip(motor_vels, grating_vels, motor_directions, grating_directions, static_grating_intervals)):
        if vm == motor_vel and vg == grating_vel and dm == motor_direction and dg == grating_direction and sgi == static_grating_interval:

            k1 = start_inds[i]
            k2 = stop_inds[i]

            dir_trial = x[k1-n*fps:k2+n*fps]
            t_trials.append(t[k1-n*fps:k2+n*fps] - t[k1])
            trial_id.append(i)
            dir_trial = np.unwrap(dir_trial)
            dir_before = np.mean(dir_trial[(n-1)*fps:n*fps])
            dir_after = np.mean(dir_trial[-n*fps:-(n-1)*fps])
            direction_trials.append(dir_trial)
            before.append(dir_before)
            after.append(dir_after)
    
    return {'motor_velocity': motor_vel, 'grating_velocity': grating_vel, 'motor_direction': motor_direction, 'grating_direction': grating_direction, 'x': direction_trials, 't': t_trials, 'before': before, 'after': after, 'static_grating_interval': static_grating_interval, 'trial_id': trial_id}


def detect_quickphases_dir(t, p, v, direction, detection_thr, duration_thr, max_dur, mask=None):
    """
    Detects quick phases in a given direction based on velocity and duration thresholds.

    Parameters:
    t: array-like
        Time array.
    p: array-like
        Eye position array.
    v: array-like
        Eye velocity array.
    direction: int
        Direction of quick phases to detect (1 for positive, -1 for negative).
    detection_thr: float
        Velocity threshold for quick phase detection.
    duration_thr: float
        Minimum duration threshold for quick phases.
    max_dur: int
        Maximum duration for a valid quick phase.
    mask: array-like, optional
        Mask array indicating already detected quick phases. Default is None.

    Returns:
    ind: list of arrays
        Indices of detected quick phases.
    ti: list of floats
        Start times of detected quick phases.
    tf: list of floats
        End times of detected quick phases.
    t_qp: list of arrays
       Time arrays for each detected quick phases.
    amplitude: list of floats
        Amplitudes of detected quick phases.
    direction: list of ints
        Directions of detected quick phases.
    mask: array-like
        Updated quick phase mask.
    """

    ind = []
    ti = []
    tf = []
    t_qp = []
    amplitude = []
    directions = []

    if mask is None:
        mask = np.zeros_like(t, dtype=bool)

    duration_mask = (direction * v) > duration_thr
    labeled, n_intervals = label(duration_mask)

    for s in range(1, n_intervals + 1):

        mask_interval = labeled == s

        qp_ind = np.where(mask_interval)[0]
        last_ind = qp_ind[-1]
        last_ind += (last_ind + 1 < len(mask_interval))
        mask_interval[last_ind] = True

        v_interval = v[mask_interval]
        max_vel = np.max(direction * v_interval)

        if max_vel > detection_thr and qp_ind[-1] - qp_ind[0] <= max_dur:

            if np.sum(mask_interval * mask) == 0:

                mask[mask_interval] = True
                t_interval = t[mask_interval]
                p_interval = p[mask_interval]
                t_qp.append(t_interval)
                ind.append(np.where(mask_interval)[0])
                ti.append(t_interval[0])
                tf.append(t_interval[-1])
                amp = np.abs(p_interval[-1] - p_interval[0])
                amplitude.append(amp)
                directions.append(direction)

    return ind, ti, tf, t_qp, amplitude, directions, mask


def detect_quickphases(t, p, v, fps=60):
    """
    Detects quick phases in both directions based on velocity and duration thresholds.

    Parameters
    ----------
    t : array-like
        Time array.
    p : array-like
        Eye position array.
    v : array-like
        Eye velocity array.
    fps : int, optional
        Frames per second. Default is 60.

    Returns
    -------
    dict
        Dictionary containing detected quick phases data with the following keys:
        - 't' : list of arrays
            Time arrays for each detected quick phase.
        - 'ind' : list of arrays
            Indices of detected quick phases.
        - 'ti' : array of floats
            Start times of detected quick phases.
        - 'tf' : array of floats
            End times of detected quick phases.
        - 'amplitude' : array of floats
            Amplitudes of detected quick phases.
        - 'direction' : array of ints
            Directions of detected quick phases.
    """

    detection_thr = 150 * np.pi / 180
    duration_thr = 10 * np.pi / 180
    max_dur = 12

    # first find quick phases in the direction opposite to the mean eye direction
    mean_eye_dir = np.sign(np.nanmean(np.sign(v)))
    qp_dir = -mean_eye_dir

    ind, ti, tf, t_qp, amplitude, direction, qp_mask = detect_quickphases_dir(t, p, v, qp_dir, detection_thr, duration_thr, max_dur, mask=None)

    from skimage.morphology import binary_dilation

    exclude = (100 * 60) // 1000
    qp_mask = binary_dilation(qp_mask, footprint=np.ones(2*exclude+1))
    qp_dir *= -1

    # now find quick phases in the opposite direction, under the condition that there is no other quick phase close in time
    ind2, ti2, tf2, t_qp2, amplitude2, direction2, _ = detect_quickphases_dir(t, p, v, qp_dir, detection_thr, duration_thr, max_dur, mask=qp_mask)

    ind += ind2
    ti += ti2
    tf += tf2
    t_qp += t_qp2
    amplitude += amplitude2
    direction += direction2

    data = {'t': t_qp, 'ind': ind, 'ti': np.array(ti), 'tf': np.array(tf), 'amplitude': np.array(amplitude), 'direction': np.array(direction)}

    return data


def fit_eye_velocity_curve(t, eye_vel, start_inds, stop_inds, fps=60, deg=4, t0=0, tf=3):
    """
    Fits a polynomial curve to eye velocity data, excluding periods of quick phases.

    Parameters
    ----------
    t : array-like
        Time array.
    eye_vel : array-like
        Eye velocity array.
    start_inds : array-like
        Indices marking the start of quick phases.
    stop_inds : array-like
        Indices marking the stop of quick phases.
    fps : int, optional
        Frames per second. Default is 60.
    deg : int, optional
        Degree of the polynomial to fit. Default is 4.
    t0 : float, optional
        Start time for fitting the polynomial. Default is 0.
    tf : float, optional
        End time for fitting the polynomial. Default is 3.

    Returns
    -------
    numpy.ndarray
        Coefficients of the fitted polynomial.

    Notes
    -----
    The function excludes data during quick phases from the fitting process. 
    It uses Kernel Density Estimation (KDE) to compute sample weights inversely proportional 
    to the density of time points. The fitting is done using a Huber Regressor to be 
    robust against outliers.
    """

    qp_mask = np.zeros(len(t), dtype=bool)
    for start_ind, stop_ind in zip(start_inds, stop_inds):
        qp_mask[start_ind:stop_ind + 1] = True

    y = eye_vel[~qp_mask]
    t = t[~qp_mask]
    t = t.reshape(len(t), 1)

    kde = KernelDensity(kernel="gaussian", bandwidth=0.15).fit(t)
    dens = np.exp(kde.score_samples(t))
    sample_weight = 1 / dens

    mask = np.logical_and(t > t0, t < tf).flatten()
    t = t[mask]
    y = y[mask]
    sample_weight = sample_weight[mask]

    poly = PolynomialFeatures(deg, include_bias=True)
    X = poly.fit_transform(t)
    reg = HuberRegressor(max_iter=500, fit_intercept=False).fit(X, y, sample_weight)

    return reg.coef_


def get_eye_data(t, p, v, start_inds, stop_inds, stim_list, stim, n=2, T=3, fps=60):
    """
    Extracts and processes eye movement data for trials that match specified stimulus conditions.

    Parameters
    ----------
    t : array-like
        Time array.
    p : array-like
        Eye position array.
    v : array-like
        Eye velocity array.
    start_inds : array-like
        Indices marking the start of trials.
    stop_inds : array-like
        Indices marking the stop of trials.
    stim_list : tuple of arrays
        Lists of stimulus parameters: motor velocities, grating velocities, motor directions, 
        grating directions, and static grating intervals.
    stim : tuple
        Specific stimulus conditions to match (motor velocity, grating velocity, motor direction,
        grating direction, static grating interval).
    n : int, optional
        Number of seconds to include before and after each trial. Default is 2.
    T : int, optional
        Total duration of the trial in seconds. Default is 3.
    fps : int, optional
        Frames per second. Default is 60.

    Returns
    -------
    dict
        Dictionary containing processed eye movement data with the following keys:
        - 'motor_velocity': Motor velocity for the trials.
        - 'grating_velocity': Grating velocity for the trials.
        - 'motor_direction': Motor direction for the trials.
        - 'grating_direction': Grating direction for the trials.
        - 'eye_position': List of arrays with eye position data for each trial.
        - 'eye_velocity': List of arrays with eye velocity data for each trial.
        - 't_eye': List of arrays with time data for each trial.
        - 'eye_pos_before': List of mean eye positions before each trial.
        - 'eye_pos_after': List of mean eye positions after each trial.
        - 'ti_qp': List of arrays with start times of quick phases for each trial.
        - 'tf_qp': List of arrays with end times of quick phases for each trial.
        - 'qp_amplitude': List of arrays with amplitudes of quick phases for each trial.
        - 'qp_direction': List of arrays with directions of quick phases for each trial.
        - 'eye_reliable': List of booleans indicating the reliability of eye velocity data for each trial.
        - 'eye_velocity_coef': List of coefficients from polynomial fits of eye velocity for each trial.
        - 'eye_velocity_coef_2nd_half': List of coefficients from polynomial fits of eye velocity for the second half of each trial.
        - 'static_grating_interval': Static grating interval for the trials.

    Notes
    -----
    This function processes eye movement data for trials that match the specified stimulus conditions.
    It excludes periods of quick phases, smooths the remaining data, and fits polynomial curves
    to the eye velocity data. The function also calculates the reliability of the eye velocity data
    and the median eye velocity during non-quick phase periods.
    """

    motor_vels, grating_vels, motor_directions, grating_directions, static_grating_intervals = stim_list
    motor_vel, grating_vel, motor_direction, grating_direction, static_grating_interval = stim

    eye_position = []
    eye_velocity = []
    t_trials = []
    before = []
    after = []
    ti_qp = []
    tf_qp = []
    qp_amplitude = []
    qp_direction = []
    reliable = []
    fit_params = []
    fit_params_second_half = []

    for i, (vm, vg, dm, dg, sgi) in enumerate(zip(motor_vels, grating_vels, motor_directions, grating_directions, static_grating_intervals)):

        if vm == motor_vel and vg == grating_vel and dm == motor_direction and dg == grating_direction and sgi == static_grating_interval:

            k1 = start_inds[i]
            k2 = stop_inds[i]

            eye_pos_trial = p[k1-n*fps:k2+n*fps]
            eye_vel_trial = v[k1-n*fps:k2+n*fps]

            t_trial = t[k1-n*fps:k2+n*fps] - t[k1]
            t_trials.append(t_trial)
            before.append(p[k1])
            after.append(p[k2])

            delta = fps // 4
            qp_data = detect_quickphases(t[k1-delta:k2+delta+1], p[k1-delta:k2+delta+1], v[k1-delta:k2+delta+1])

            t_start = qp_data['ti'] - t[k1]
            t_stop = qp_data['tf'] - t[k1]

            ti_qp.append(t_start)
            tf_qp.append(t_stop)
            qp_amplitude.append(qp_data['amplitude'])
            qp_direction.append(qp_data['direction'])

            if np.any(np.isnan(v[k1-delta:k2+delta+1])):
                reliable.append(False)
                fit_params.append([])
                fit_params_second_half.append([])
            else:
                reliable.append(True)
                if sgi == -1:
                    start_ind = [np.argmin(np.abs(t_trial - x)) for x in t_start]
                    stop_ind = [np.argmin(np.abs(t_trial - x)) for x in t_stop]
                    coef = fit_eye_velocity_curve(t_trial, eye_vel_trial, start_ind, stop_ind)
                    fit_params.append(coef)
                    fit_params_second_half.append([])
                else:

                    exit()
                    # start_ind = [np.argmin(np.abs(t_trial - x)) for x in t_start]
                    # stop_ind = [np.argmin(np.abs(t_trial - x)) for x in t_stop]
                    # coef = fit_eye_velocity_curve(t_trial, eye_vel_trial, start_ind, stop_ind, deg=2, t0=0, tf=1.5)
                    # fit_params.append(coef)
                    # coef2 = fit_eye_velocity_curve(t_trial, eye_vel_trial, start_ind, stop_ind, deg=2, t0=1.5, tf=3.0)
                    # fit_params_second_half.append(coef2)

            eye_position.append(eye_pos_trial)
            eye_velocity.append(eye_vel_trial)

    return {'motor_velocity': motor_vel, 'grating_velocity': grating_vel, 'motor_direction': motor_direction, 'grating_direction': grating_direction,
            'eye_position': eye_position, 'eye_velocity': eye_velocity, 't_eye': t_trials, 'eye_pos_before': before, 'eye_pos_after': after,
            'ti_qp': ti_qp, 'tf_qp': tf_qp, 'qp_amplitude': qp_amplitude, 'qp_direction': qp_direction, 'eye_reliable': reliable,
            'eye_velocity_coef': fit_params, 'eye_velocity_coef_2nd_half': fit_params_second_half, 'static_grating_interval': static_grating_interval}


def interpolate_filter_eye_pos(t_eye, eye_pos):
    """
    Interpolates and filters eye position data, filling NaN values and smoothing the result.

    Parameters
    ----------
    t_eye : array-like
        Time array corresponding to the eye position data.
    eye_pos : array-like
        Eye position array with potential NaN values to be interpolated.

    Returns
    -------
    numpy.ndarray
        Eye position array with NaN values interpolated and the data smoothed using a Gaussian filter.

    Notes
    -----
    This function interpolates NaN values in the eye position data if both neighboring values are finite.
    The interpolation is followed by a Gaussian filter to smooth the eye position data.
    """

    # interpolate nan values if two neighboring values are finite
    nan_mask = np.isnan(eye_pos)
    interp_mask = nan_mask.copy()
    interp_mask[0] = False
    interp_mask[-1] = False
    finite_neigh = np.zeros_like(interp_mask)
    finite_neigh[1:-1] = np.logical_and(np.isfinite(eye_pos[2:]), np.isfinite(eye_pos[:-2]))
    interp_mask[~finite_neigh] = False
    try:
        interp_eye_pos = np.interp(t_eye[interp_mask], t_eye[~nan_mask], eye_pos[~nan_mask])
        eye_pos[interp_mask] = interp_eye_pos
    except ValueError:
        pass

    eye_pos = gaussian_filter1d(eye_pos, 1.0, truncate=2, mode='nearest')
    return eye_pos


def start_stop_inds(t, t_start, t_stop):
    """
    Finds the indices in a time array corresponding to specified start and end times.

    Parameters
    ----------
    t : array-like
        Time array.
    t_start : array-like
        Array of start times.
    t_stop : array-like
        Array of stop times.

    Returns
    -------
    start_inds : list of int
        List of indices in `t` where the values are just greater than each `t_start` time.
    stop_inds : list of int
        List of indices in `t` where the values are just greater than each `t_stop` time.
    """

    start_inds = [np.where(t > ti)[0] for ti in t_start]
    stop_inds = [np.where(t > tf)[0] for tf in t_stop]
    start_inds = [x[0] for x in start_inds if len(x) > 0]
    stop_inds = [x[0] for x in stop_inds if len(x) > 0]
    return start_inds, stop_inds


def load_eye_data(animal, recording):
    """
    Loads and processes eye movement data for a specified animal and recording.

    Parameters
    ----------
    animal : str
    recording : str

    Returns
    -------
    t_eye : numpy.ndarray
        Time array for the eye data.
    eye_pos : numpy.ndarray
        Interpolated and filtered eye position array.
    eye_vel : numpy.ndarray
        Eye velocity array computed from the eye position.
    start_inds_eye : list of int
        Indices in the time array where trials start.
    stop_inds_eye : list of int
        Indices in the time array where trials stop.
    """

    with open(os.path.join(BASE_PATH, 'data', 'base_data', '{}.p'.format(recording)), 'rb') as file:
        data = pickle.load(file)
    eye_pos = EYE_ORIENTATION * data['eye_pos']
    t_eye = data['t_eye']

    if len(eye_pos) > 0:
        eye_pos = interpolate_filter_eye_pos(t_eye, eye_pos)
        fps_eye = 60
        eye_vel = fps_eye * np.diff(eye_pos, append=np.nan)
    else:
        eye_vel = np.array([])

    t_start = data['stimulus_start']
    t_stop = data['stimulus_stop']
    start_inds_eye, stop_inds_eye = start_stop_inds(t_eye, t_start, t_stop)

    return t_eye, eye_pos, eye_vel, start_inds_eye, stop_inds_eye


def load_data(animal, recording, eyetracking=False, dark=False, stage=0, extract=DECODED_DIRECTION, adjust=True):
    """
    Loads and processes experimental data for a specified animal and recording session.

    Parameters
    ----------
    animal : str
        Identifier for the animal.
    recording : str
        Identifier for the recording session.
    eyetracking : bool, optional
        Flag to include eye tracking data. Default is False.
    dark : bool, optional
        Flag to indicate dark condition. Default is False.
    stage : int, optional
        Stage of the experiment to load. Default is 0.
    extract : str, optional
        The data to extract (DECODED_DIRECTION or 'hd'). Default is DECODED_DIRECTION.
    adjust : bool, optional
        Flag to adjust time arrays. Default is True.

    Returns
    -------
    data : list of dict
        Processed data for each stimulus condition, including decoded directions and optionally eye tracking data.
    stimset : str
        Type of stimulus set used in the experiment.
    T : int
        Duration of the trial.
    eyetracking : bool
        Flag indicating whether eye tracking data is included.
    """

    experiment_path = os.path.join(BASE_PATH, 'data', 'stim_sets', recording)
    with open(os.path.join(experiment_path, 'stim_set_type.txt'), 'r') as file:
        stimset = file.read()

    motor_vels, motor_directions, grating_vels, grating_directions = load_stimuli(experiment_path, STIM_ORIENTATION)

    if stimset == 'asymmetric':
        with np.load(os.path.join(experiment_path, 'static_grating_interval.npz')) as npz:
            static_grating_intervals = np.ma.MaskedArray(**npz)
        static_grating_intervals.data[static_grating_intervals.mask] = -1
        static_grating_intervals = static_grating_intervals.data
    else:
        static_grating_intervals = -np.ones_like(grating_directions)

    stims = set(zip(motor_vels, grating_vels, motor_directions, grating_directions, static_grating_intervals))

    with open(os.path.join(BASE_PATH, 'data', 'embedding_data', '{}.p'.format(recording)), 'rb') as file:
        data = pickle.load(file)

    stage_mask = data['stage'] == stage
    t = data['time'][stage_mask]
    if extract == DECODED_DIRECTION:
        x = data[DECODED_DIRECTION][stage_mask]
    elif extract == 'hd':
        x = data['hd'][stage_mask]

    t_start = data['stimulus_start']
    t_stop = data['stimulus_stop']

    if adjust:
        t -= DECODED_PHYSICAL_LAG
        t_start = [x - DECODED_PHYSICAL_LAG for x in t_start]
        t_stop = [x - DECODED_PHYSICAL_LAG for x in t_stop]

    start_inds, stop_inds = start_stop_inds(t, t_start, t_stop)

    # only one stage for eye tracking experiments
    if eyetracking:
        t_eye, eye_pos, eye_vel, start_inds_eye, stop_inds_eye = load_eye_data(animal, recording)
        if len(eye_pos) == 0:
            eyetracking = False

    data = []
    stim_list = (motor_vels, grating_vels, motor_directions, grating_directions, static_grating_intervals)
    for stim in stims:

        data_stim = extract_trials(t, x, start_inds, stop_inds, stim_list, stim)

        if extract == DECODED_DIRECTION:
            data_stim[DECODED_DIRECTION] = data_stim.pop("x")
        else:
            data_stim["hd"] = data_stim.pop("x")

        if eyetracking:
            keys = ['eye_position', 'eye_velocity', 't_eye', 'eye_pos_before', 'eye_pos_after', 'ti_qp', 'tf_qp', 'qp_amplitude', 'qp_direction', 'eye_reliable', 'eye_velocity_coef', 'eye_velocity_coef_2nd_half']
            data_eye = get_eye_data(t_eye, eye_pos, eye_vel, start_inds_eye, stop_inds_eye, stim_list, stim, T=T_TRANSIENT_STIM)
            for key in keys:
                data_stim[key] = data_eye[key]

        data.append(data_stim)

    return data, stimset, eyetracking


def create_df(all_data, v_max, animal, recording, eyetracking=False, T=3, dark=False, var=DECODED_ROTATION):
    """
    Creates a DataFrame summarizing trial data, including eye tracking information if available.

    Parameters
    ----------
    all_data : list of dict
        List of dictionaries containing processed trial data.
    v_max : dict
        Dictionary mapping velocity indices to maximum velocities.
    animal : str
        Identifier for the animal.
    recording : str
        Identifier for the recording session.
    eyetracking : bool, optional
        Flag to include eye tracking data. Default is False.
    T : int, optional
        Total duration of the trial in seconds. Default is 3.
    dark : bool, optional
        Flag to indicate dark condition. Default is False.
    var : str, optional
        The variable to decode and include in the DataFrame. Default is DECODED_ROTATION.

    Returns
    -------
    pandas.DataFrame
        DataFrame summarizing the trial data, including total rotations velocities and eye movements.
    """

    grating_vels = []
    motor_vels = []
    rotations = []
    neg_eye_rotation = []
    qp_amplitude = []
    qp_count = []
    eye_velocity_coef = []
    eye_velocity_coef_2nd_half = []
    sgi = []
    trial_id = []

    for data in all_data:
        after = np.array(data['after'])
        before = np.array(data['before'])

        rots_data = after - before
        rotations += rots_data.tolist()
        trial_id += data['trial_id']

        n_trials = len(rots_data)
        grating_vels += [data['grating_direction'] * v_max[data['grating_velocity']]] * n_trials
        motor_vels += [data['motor_direction'] * v_max[data['motor_velocity']]] * n_trials
        sgi += [data['static_grating_interval']] * n_trials

        if eyetracking:
            eye_velocity_coef += data['eye_velocity_coef']
            eye_velocity_coef_2nd_half += data['eye_velocity_coef_2nd_half']
            neg_eye_rotation += [-comp_eye_rotation(x1, x2) for x1, x2 in zip(data['eye_velocity_coef'], data['eye_velocity_coef_2nd_half'])]

            for ti_qp, tf_qp, qp_amp, qp_dir, reliable in zip(data['ti_qp'], data['tf_qp'], data['qp_amplitude'], data['qp_direction'], data['eye_reliable']):

                if reliable:
                    mask = np.logical_and(ti_qp > 0, tf_qp < T)
                    qp_amplitude.append(np.sum(qp_dir[mask] * qp_amp[mask]))
                    qp_count.append(np.sum(mask))
                else:
                    qp_amplitude.append(np.nan)
                    qp_count.append(np.nan)

    df = pd.DataFrame()
    df['grating_velocity'] = grating_vels
    df['motor_velocity'] = motor_vels
    df['trial_id'] = trial_id
    df[var] = rotations
    df['dark'] = dark
    df['static_grating_interval'] = sgi
    if eyetracking:
        df['quick_phase_amplitude'] = qp_amplitude
        df['quick_phase_count'] = qp_count
        df['eye_velocity_coef'] = eye_velocity_coef
        df['eye_velocity_coef_2nd_half'] = eye_velocity_coef_2nd_half
        df[NEG_EYE_ROTATION] = neg_eye_rotation
    else:
        df['quick_phase_amplitude'] = np.nan
        df['quick_phase_count'] = np.nan
        df['eye_velocity_coef'] = np.empty((len(df), 0)).tolist()
        df['eye_velocity_coef_2nd_half'] = np.empty((len(df), 0)).tolist()
        df[NEG_EYE_ROTATION] = np.nan

    df['animal'] = animal
    df['recording'] = recording

    return df


def plot_trials(data, v_max, fig_path, T=3, stimset='all_combinations', orientation=1, plot_eye=True):

    os.makedirs(os.path.join(fig_path, 'unaligned'), exist_ok=True)
    os.makedirs(os.path.join(fig_path, 'aligned'), exist_ok=True)
    os.makedirs(os.path.join(fig_path, 'overlayed'), exist_ok=True)
    os.makedirs(os.path.join(fig_path, 'eye_pos_and_rotation'), exist_ok=True)

    for data_stim in data:
        dm = data_stim['motor_direction']
        dg = data_stim['grating_direction']
        vm = dm * v_max[data_stim['motor_velocity']]
        vg = dg * v_max[data_stim['grating_velocity']]
        sgi = data_stim['static_grating_interval']

        title = HEAD_VAR + ' = {} deg / s, '.format(vm) + SCENE_VAR + ' = {} deg / s'.format(vg)
        if sgi == -1:
            fname = 'motor={}_grating={}.pdf'.format(vm, vg)
        else:
            fname = 'motor={}_grating={}_asym={}.pdf'.format(vm, vg, sgi)

        visualization.plot_unaligned_traces(data_stim['t'], data_stim[DECODED_DIRECTION], title=title, filename=fname, figpath=fig_path, T=T)

        if 'eye_position' in data_stim and plot_eye:

            if sgi == -1:
                fig_path_eye = os.path.join(fig_path, 'eye_pos_and_rotation', 'motor={}_grating={}'.format(vm, vg))
            else:
                fig_path_eye = os.path.join(fig_path, 'eye_pos_and_rotation', 'motor={}_grating={}_asym={}'.format(vm, vg, sgi))
            os.makedirs(fig_path_eye, exist_ok=True)

            visualization.plot_eyepos_and_rotation(data_stim, title=title, filename=fname, figpath=fig_path_eye, T=3, orientation=orientation)

        visualization.plot_aligned_traces(data_stim['t'], data_stim[DECODED_DIRECTION], data_stim['before'], title=title, filename=fname, figpath=fig_path, T=T)

    data_catch = [data_stim for data_stim in data if data_stim['motor_velocity']==0 and data_stim['grating_velocity']==0]
    if len(data_catch) > 0:
        data_catch = data_catch[0]
    else:
        data_catch = None

    for (data1, data2) in itertools.product(data, data):

        if data1['motor_direction'] == -data2['motor_direction'] and data1['grating_direction'] == -data2['grating_direction'] and data1['motor_velocity'] == data2['motor_velocity'] and data1['grating_velocity'] == data2['grating_velocity'] and not (data1['motor_velocity'] == 0 and data1['grating_velocity'] == 0) and (data1['static_grating_interval'] == data2['static_grating_interval']):

            vm1 = data1['motor_direction'] * v_max[data1['motor_velocity']]
            vg1 = data1['grating_direction'] * v_max[data1['grating_velocity']]
            sgi = data1['static_grating_interval']

            if vm1 < 0 or (vm1 == 0 and vg1 < 0):
                continue

            if vg1 > 0:
                title = HEAD_VAR + ' = ' + r'$\pm$' + '{} deg / s, '.format(vm1) + SCENE_VAR + ' = ' + r'$\pm$' + '{} deg / s'.format(vg1)
            elif vg1 < 0:
                title = HEAD_VAR + ' = ' + r'$\pm$' + '{} deg / s, '.format(vm1) + SCENE_VAR + ' = ' + r'$\pm$' + '{} deg / s'.format(-vg1)
            else:
                title = HEAD_VAR + ' = ' + r'$\pm$' + '{} deg / s, '.format(vm1) + SCENE_VAR + ' = 0 deg / s'

            if sgi == -1:
                fname = 'motor1={}_grating1={}_motor2={}_grating2={}.pdf'.format(vm1, vg1, -vm1, -vg1)
            else:
                fname = 'motor1={}_grating1={}_motor2={}_grating2={}_asym={}.pdf'.format(vm1, vg1, -vm1, -vg1, sgi)

            t, x = get_head_trajectory(vm1, T)
            visualization.overlay_aligned_traces(data1, data2, data_catch=data_catch, title=title, filename=fname, figpath=fig_path, t_theo=t, x_theo=x, T=T)


def get_orientation(df):
    """
    Determines the orientation of the decoded direction based on the correlation between decoded rotation and motor velocity.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the trial data, including decoded rotation and motor velocity.

    Returns
    -------
    int
        Orientation of the stimulus. Returns 1 if the correlation between decoded rotation and motor velocity is positive,
        otherwise returns -1.
    """

    mask = df['stim_type'] == PLATFORM_STR
    rho = np.corrcoef(df[mask][DECODED_ROTATION], df[mask]['motor_velocity'])[1,0]
    orientation = 2 * (rho > 0) - 1

    return orientation


def get_orientation_asymmetric(df):
    """
    Determines the orientation of the decoded direction for asymmetric stimulus sets based on the correlation between decoded rotation and motor velocity.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the trial data, including decoded rotation, motor velocity, and grating velocity.

    Returns
    -------
    int
        Orientation of the stimulus. Returns 1 if the correlation between decoded rotation and motor velocity is positive,
        otherwise returns -1.
    """

    mask = np.logical_or(df['motor_velocity'].abs() > 0, np.logical_and(df['motor_velocity'] == 0, df['grating_velocity'] == 0))
    rho = np.corrcoef(df[DECODED_ROTATION][mask], df['motor_velocity'][mask])[1,0]
    orientation = 2 * (rho > 0) - 1
    return orientation


def main(animal, recording, eyetracking, stage=0, dark=False):

    data, stimset, eyetracking = load_data(animal, recording, eyetracking=eyetracking, dark=dark, extract=DECODED_DIRECTION, stage=stage)

    data_path = os.path.join(BASE_PATH, 'data', 'trial_data', recording)
    os.makedirs(data_path, exist_ok=True)

    v_max = get_vmax(stimset)

    if stimset == 'asymmetric':
        save_trial_data(data_path, data, v_max, asymmetric=True)
    else:
        save_trial_data(data_path, data, v_max)

    df = create_df(data, v_max, animal, recording, eyetracking=eyetracking, dark=dark, T=T_TRANSIENT_STIM)
    df_restricted = extract_restricted_data(df, dark=dark)

    if stimset == 'active_grating':
        orientation = 1  # already oriented
    elif stimset == 'asymmetric':
        orientation = get_orientation_asymmetric(df)
    else:
        orientation = get_orientation(df_restricted)

    df_restricted[DECODED_ROTATION] *= orientation

    with open(os.path.join(data_path, 'orientation.txt'), 'w') as f:
        f.write(str(orientation))

    fig_path = os.path.join(BASE_PATH, 'figures', recording)
    os.makedirs(fig_path, exist_ok=True)

    plot_trials(data, v_max, fig_path, T=T_TRANSIENT_STIM, stimset=stimset, orientation=orientation)

    csvname = 'restricted_stimset.csv'
    df_restricted.to_csv(os.path.join(data_path, csvname), index=False)

    pklname = 'restricted_stimset.p'
    df_restricted.to_pickle(os.path.join(data_path, pklname))

    df[DECODED_ROTATION] *= orientation

    csvname = 'agg_data.csv'
    df.to_csv(os.path.join(data_path, csvname), index=False)

    pklname = 'agg_data.p'
    df.to_pickle(os.path.join(data_path, pklname))

    if stimset == 'active_grating':
        data_hd = load_data(animal, recording, eyetracking=eyetracking, dark=False, extract='hd')[0]
        df_hd = create_df(data_hd, v_max, animal, recording, eyetracking=eyetracking, dark=False, var=TRUE_ROTATION)
        df_hd = df_hd[['trial_id', TRUE_ROTATION]]

        df_join = df.join(df_hd.set_index('trial_id'), on='trial_id', lsuffix='_decoded', rsuffix='_hd')
        df_join['delta_rotation'] = df_join[DECODED_ROTATION] - df_join[TRUE_ROTATION]

        pklname = 'active_grating_data.p'
        df_join.to_pickle(os.path.join(data_path, pklname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='analyze transient stimuli')
    parser.add_argument('-a', '--animal', required=True)
    parser.add_argument('-r', '--recording', required=True)
    parser.add_argument('-e', '--eyetracking', action='store_true')
    parser.add_argument('-d', '--dark', action='store_true')
    parser.add_argument('-s', '--stage', default=0, type=int, required=False)
    args = parser.parse_args()

    main(args.animal, args.recording, args.eyetracking, dark=args.dark, stage=args.stage)
