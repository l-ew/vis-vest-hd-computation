from scipy import stats
from vis_vest_hd_computation.imports import *
import vis_vest_hd_computation.visualization as visualization
import vis_vest_hd_computation.utils as utils


def main(recording, recording_type):

    with open(os.path.join(BASE_PATH, 'data', 'embedding_data', '{}.p'.format(recording)), 'rb') as file:
        data = pickle.load(file, encoding="latin1")

    fig_path = os.path.join(BASE_PATH, 'figures', recording, 'other')
    os.makedirs(fig_path, exist_ok=True)

    deconvolved_act = data['deconvolved_act']
    decoded_direction = data[DECODED_DIRECTION]
    latent_state = data['latent_state']
    hd = data['hd']
    stage = data['stage'].astype(int)
    t = data['time']

    drift = utils.comp_delta(decoded_direction, hd)

    hd_bins = 36
    hd_bin_edges = np.linspace(-np.pi, np.pi, hd_bins + 1)
    hd_bin_centers = 2 * np.pi * (np.arange(hd_bins) + 0.5) / hd_bins - np.pi

    stages = np.unique(stage)
    decoding_error = []
    vec_lengths = []
    peak_firing_dirs = []
    latent_vec_lengths = []
    latent_peak_firing_dirs = []
    significant_hd_tuning = []
    significant_latent_tuning = []
    n_cells = []
    circ_corrs = []

    if recording_type == 'combined':
        t[stage == 1] += t[stage == 0][-1] + 1.0 / FPS_MINISCOPE

    for k in stages:
        mask = stage == k

        # mask2 = np.logical_and(mask, np.logical_not(np.isnan(hd)))
        # rho = utils.circ_corr_mem(latent_state[mask2], hd[mask2])

        n_cells.append(deconvolved_act.shape[1])

        mad = np.nanmedian(np.abs(drift[mask]))
        decoding_error.append(mad)

        tuning_curves, _ = utils.comp_tuning_curves(hd[mask], deconvolved_act[mask,:], hd_bin_edges)
        _, vec_length_stage = utils.rayleigh_vector(tuning_curves, hd_bin_centers)

        vec_lengths.append(vec_length_stage)
        randomized_vec_lengths = gen_sample(deconvolved_act[mask,:], hd[mask], hd_bin_edges, random=False, fps=FPS_MINISCOPE)
        perc = [np.percentile(randomized_vec_lengths[i,:], 99.0, interpolation='linear') for i in range(n_cells[k])]
        significant_hd_tuning.append(vec_length_stage > perc)

        peak_firing_dir = utils.comp_peak_firing_direction(tuning_curves, hd_bin_centers)[0]
        peak_firing_dirs.append(peak_firing_dir)

        latent_tuning_curves, _ = utils.comp_tuning_curves(decoded_direction[mask], deconvolved_act[mask,:], hd_bin_edges)
        _, latent_vec_length_stage = utils.rayleigh_vector(latent_tuning_curves, hd_bin_centers)
        latent_vec_lengths.append(latent_vec_length_stage)
        latent_peak_firing_dir = utils.comp_peak_firing_direction(latent_tuning_curves, hd_bin_centers)[0]
        latent_peak_firing_dirs.append(latent_peak_firing_dir)

        randomized_latent_vec_lengths = gen_sample(deconvolved_act[mask,:], decoded_direction[mask], hd_bin_edges, random=True, n_reps=10, fps=FPS_MINISCOPE)
        perc = [np.percentile(randomized_latent_vec_lengths[i,:], 99.0, interpolation='linear') for i in range(n_cells[k])]
        significant_latent_tuning.append(latent_vec_length_stage > perc)

        lag, circ_corr = calc_circ_correlation(hd[mask], decoded_direction[mask], FPS_MINISCOPE)
        circ_corrs.append(circ_corr)

    decoding_error = np.array(decoding_error)

    data = {'decoding_error': decoding_error, 'vec_length': vec_lengths, 'peak_firing_dir': peak_firing_dirs, 'significant_hd_tuning': significant_hd_tuning,
            'n_cells': n_cells, 'circ_corr': circ_corrs, 'latent_vec_length': latent_vec_lengths,
            'latent_peak_firing_dir': latent_peak_firing_dirs, 'significant_latent_tuning': significant_latent_tuning}

    return data


def gen_sample(activities, hd_trace, hd_bin_edges, fps=12, n_reps=1000, random=True):
    """
    Generate samples of vector lengths based on activities and lagged head direction traces.

    This function generates samples of vector lengths by shifting the head direction (`hd_trace`) in time
    and computing the tuning curves of the `activities` with respect to the shifted head direction. 

    Parameters:
    activities : numpy.ndarray
        The neural activity data. Shape should be (n_time_points, n_cells).
    hd_trace : numpy.ndarray
        The head direction trace. Shape should be (n_time_points,).
    hd_bin_edges : numpy.ndarray
        The edges of the bins for head direction. Shape should be (n_bins + 1,).
    fps : int, optional
        The frames per second of the data. Default is 12.
    n_reps : int, optional
        The number of repetitions for generating samples. Default is 1000.
    random : bool, optional
        A flag indicating whether to use random shifts. If False, uses sequential shifts. Default is True.

    Returns:
    numpy.ndarray
        The vector lengths for each cell and each repetition. Shape is (n_cells, n_reps).
    """
    min_shift = 5 * fps
    max_shift = 60 * fps

    if not random:
        n_reps = max_shift - min_shift

    n_cells = activities.shape[1]
    vec_lengths = np.empty((n_cells, n_reps))
    hd_bins = len(hd_bin_edges) - 1
    hd_bin_centers = 2 * np.pi * (np.arange(hd_bins) + 0.5) / hd_bins - np.pi

    for i in range(n_reps):
        if random:
            shifted_hd_trace = np.roll(hd_trace, np.random.randint(-max_shift, -min_shift))
        else:
            shifted_hd_trace = np.roll(hd_trace, -i-min_shift)

        tuning_curves = stats.binned_statistic(shifted_hd_trace, [activities[:,i] for i in range(n_cells)], statistic='mean', bins=hd_bin_edges)[0]
        norm_const = np.nansum(tuning_curves, axis=1)
        tuning_curves = tuning_curves / norm_const[:, np.newaxis]
        vec_lengths[:,i] = utils.rayleigh_vector(tuning_curves, hd_bin_centers)[1]

    return vec_lengths


def calc_circ_correlation(hd, decoded_direction, fps):
    """
    Calculate the circular correlation between head direction and decoded direction with time lags.

    Parameters:
    hd : numpy.ndarray
        The head direction time series. Shape should be (n_time_points,).
    decoded_direction : numpy.ndarray
        The decoded direction time series. Shape should be (n_time_points,).
    fps : int
        The frames per second of the data.

    Returns:
    tuple
        lags : numpy.ndarray
            The time lags for which the circular correlation is computed.
        rho : numpy.ndarray
            The circular correlation values for each lag.
    """
    n_lags = 2 * fps
    lags = np.arange(-n_lags, n_lags+1) / fps

    decoded_head_directions = []
    hds = []

    for delay in range(-n_lags, n_lags+1):
        if delay > 0:
            x = hd[:-delay].copy()
            y = decoded_direction[delay:].copy()
        elif delay < 0:
            x = hd[-delay:].copy()
            y = decoded_direction[:delay].copy()
        else:
            x = hd.copy()
            y = decoded_direction.copy()

        mask = np.logical_not(np.isnan(x))
        decoded_head_directions.append(y[mask])
        hds.append(x[mask])

    from joblib import Parallel, delayed
    rho = Parallel(n_jobs=n_lags)(delayed(utils.circ_corr_mem)(x, y) for x, y in zip(hds, decoded_head_directions))

    # rho = [np.nan] * (2 * n_lags + 1)

    return lags, rho