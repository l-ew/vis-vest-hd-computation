from sklearn.manifold import SpectralEmbedding
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.preprocessing import quantile_transform
from sklearn.metrics import pairwise_distances
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.stats import circmean
from scipy.spatial import Delaunay
import weighted
import cmocean
from vis_vest_hd_computation.imports import *
from vis_vest_hd_computation.utils import comp_tuning_curves, normalize_tuning, circ_corr, comp_delta


def cart2pol(x, y):
    """
    Convert Cartesian coordinates to polar coordinates.

    Parameters:
    x : float or array-like
        The x-coordinate(s) in Cartesian coordinates.
    y : float or array-like
        The y-coordinate(s) in Cartesian coordinates.

    Returns:
    rho : float or array-like
        The radial distance(s) from the origin.
    phi : float or array-like
        The angle(s) in radians from the positive x-axis.
    """

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    """
    Convert polar coordinates to Cartesian coordinates.

    Parameters:
    rho : float or array-like
        The radial distance(s) from the origin.
    phi : float or array-like
        The angle(s) in radians from the positive x-axis.

    Returns:
    x : float or array-like
        The x-coordinate(s) in Cartesian coordinates.
    y : float or array-like
        The y-coordinate(s) in Cartesian coordinates.
    """

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def effective_neighbors(delta, eff_num_neighbors, sigma):
    """
    Compute the squared error between a desired number of neighbors and 
    an effective number of neighbors obtained by weighting distances 
    using a Gaussian kernel and a scale parameter.

    Parameters:
    delta : float or array-like
        The distance(s) between data points.
    eff_num_neighbors : float
        The desired number of neighbors to compare against.
    sigma : float
        The scale parameter used for the Gaussian kernel.

    Returns:
    float
        The squared error between the desired number of neighbors and the 
        effective number of neighbors.
    """

    weights = np.exp(-delta**2/sigma**2)
    return (np.sum(weights) - eff_num_neighbors)**2


def compare_tuning_curves(bin_centers, latent_tuning, hd_tuning=None, path='tuning_curves.pdf'):
    n_components, n_bins = latent_tuning.shape
    n_rows = 1 + (n_components-1) // 5
    figsize = (8.27, 1.75 * n_rows)
    fig, axs = plt.subplots(n_rows, 5, figsize=figsize, tight_layout=True, sharex=False, sharey=True)
    i = 0
    for ax in axs.ravel():
        if i < n_components:
            x = np.hstack([bin_centers[0] - 2*np.pi/n_bins, bin_centers, bin_centers[-1] + 2*np.pi/n_bins])
            y = latent_tuning[i,:]
            y = np.hstack([y[-1], y, y[0]])
            ax.plot(x, y, c=DECODING_COLOR)
           
            if hd_tuning is not None:
                y = hd_tuning[i,:]
                y = np.hstack([y[-1], y, y[0]])
                ax.plot(x, y, c=HEAD_COLOR)
           
            ax.set_title('Neuron # ' + str(i))
            ax.set_ylabel('Activity [a.u.]')
            ax.set_xlabel("Head direction [rad]")
            ax.set_xlim([-np.pi, np.pi])
            ax.set_ylim([0, 1.1])
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        else:
            ax.set_axis_off()
        i = i + 1

    fig.savefig(path)
    plt.close(fig)


def tuning_heatmap(tuning_curves, label=DECODED_DIRECTION, flipped=True, ordering=None, path='tuning_heatmap.pdf'):

    if ordering is None:
        max_ind = np.argmax(tuning_curves, axis=1)
        ordering = np.argsort(max_ind)

    if flipped:
        fig, ax = plt.subplots(figsize=[2, 3])
        im = ax.matshow(tuning_curves[ordering,:], cmap=ACTIVITY_COLORMAP, vmin=0, vmax=1, aspect=1)
        ax.set_ylabel('Cell # (ordered)');
        ax.set_xticks(np.linspace(-0.5, 35.5, 5))
        ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax.set_xlabel('{} [rad]'.format(label))
        ax.xaxis.set_ticks_position('bottom')
    else:
        fig, ax = plt.subplots(figsize=[3, 2])
        im = ax.matshow(tuning_curves[ordering,:].T, cmap=ACTIVITY_COLORMAP, vmin=0, vmax=1, aspect=1)
        ax.set_xlabel('Cell # (ordered)');
        ax.set_yticks(np.linspace(-0.5, 35.5, 5))
        ax.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax.set_ylabel('{} [rad]'.format(label))
        ax.xaxis.set_ticks_position('bottom')
    
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    fig.subplots_adjust(right=0.925)
    cbar_ax = fig.add_axes([0.975, 0.3, 0.025, 0.4])
    fig.colorbar(im, cax=cbar_ax, label = 'Activity [a.u.]', ticks=[0, 0.5, 1])

    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

    return ordering


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def plot_traces(t, true_hd, decoded_direction, path='comparison_true_vs_decoded.pdf'):

    n_chunks = np.ceil(t[-1] / 60).astype('int')

    all_chunks = []
    start = 0
    for i in range(n_chunks):
        end = find_nearest(t, 60*(i+1))
        all_chunks.append(list(range(start, end)))
        start = end

    n_rows = 10
    chunks_list = [all_chunks[i:i+n_rows] for i in range(0, len(all_chunks), n_rows)]
    with PdfPages(path) as pdf:
        for page, chunks in enumerate(chunks_list):

            figsize = (8.27, 11.69)
            fig, axs = plt.subplots(n_rows, 1, figsize=figsize)
            for i in range(n_rows):
                ax = axs[i]
                if i < len(chunks):

                    ax.scatter(t[chunks[i]], true_hd[chunks[i]], s=0.5, c=HEAD_COLOR)
                    ax.scatter(t[chunks[i]], decoded_direction[chunks[i]], s=0.5, c=DECODING_COLOR)

                    ax.set_xlim([page*n_rows*60 + i*60, page*n_rows*60 + (i+1)*60])
                    ax.set_xlabel('t [s]')
                    ax.set_ylabel('Direction [rad]')
                    ax.set_xticks(page*n_rows*60 + i*60 + 5 * np.arange(13))
                    ax.set_yticks(ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
                    ax.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']);
                    ax.spines['top'].set_visible(True)
                    ax.spines['right'].set_visible(True)

                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                 ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(7)

                else:
                    ax.set_axis_off()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close()


def mapping_latent_to_hd(latent_state, head_direction, latent_state_grid, sigma):
    """
    Computes the mapping from the latent state to the head direction using a Gaussian kernel and a scale parameter.

    Parameters:
    latent_state : float or array-like
        The latent state.
    head_direction : float or array-like
        The head direction(s) corresponding to the latent state.
    latent_state_grid : array-like
        A grid of latent states to which the mapping is performed.
    sigma : float
        The scale parameter (standard deviation) used for the Gaussian kernel.

    Returns:
    latent_to_hd : array-like
        The mapped head direction(s) corresponding to each latent state in the grid.
    """

    latent_to_hd = np.zeros_like(latent_state_grid)
    n_grid = len(latent_state_grid)
    for i in range(n_grid):
        delta = latent_state - latent_state_grid[i]
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        weights = np.exp(-delta**2/sigma**2)
        latent_to_hd[i] = np.angle(np.nansum(weights * np.exp(1j * head_direction)))
    return latent_to_hd


def calc_offset(latent_to_hd, nearest_idx, head_direction):
    """
    Calculate the circular mean offset between head directions and 
    decoded directions from a latent state mapping.

    Parameters:
    latent_to_hd : array-like
        The mapped head directions corresponding to latent states.
    nearest_idx : array-like
        Indices of the nearest latent states in the grid.
    head_direction : array-like
        The actual head directions.

    Returns:
    float
        The circular mean offset between the actual head directions 
        and the decoded directions.
    """

    decoded_direction = np.array([latent_to_hd[idx] for idx in nearest_idx])
    return circmean(head_direction - decoded_direction, nan_policy='omit')


def apply_reparametrization(latent_to_hd, nearest_idx, offset=None):
    """
    Apply reparametrization to decoded head directions using an optional offset.

    Parameters:
    latent_to_hd : array-like
        The mapped head directions corresponding to latent states.
    nearest_idx : array-like
        Indices of the nearest latent states in the grid.
    offset : float or None, optional
        The offset to apply to the decoded head directions (default is None).

    Returns:
    decoded_direction : array-like
        The reparametrized head directions.
    """

    decoded_direction = np.array([latent_to_hd[idx] for idx in nearest_idx])
    if offset:
        decoded_direction = decoded_direction + offset
        decoded_direction = (decoded_direction + np.pi) % (2 * np.pi) - np.pi
    return decoded_direction


def cross_validate_reparametrization(x, y, latent_state_grid, nearest_idx, invert=1):
    """
    Perform cross-validation for reparametrization of head direction decoding.

    Parameters:
    x : array-like
        The latent states.
    y : array-like
        The actual head directions.
    latent_state_grid : array-like
        A grid of latent states to which the mapping is performed.
    nearest_idx : array-like
        Indices of the nearest latent states in the grid.
    invert : int, optional
        The direction to check for monotonicity (default is 1).

    Returns:
    sigmas : array-like
        The range of sigma values tested.
    train_performance : array-like
        The training performance for each sigma and chunk.
    val_performance : array-like
        The validation performance for each sigma and chunk.
    is_monotonic : array-like
        Boolean array indicating whether the mapping is monotonic for each sigma.
    """

    n_sigma = 50
    sigmas = np.linspace(0.01, 0.5, n_sigma)

    n_chunks = 5
    x_chunks = np.array_split(x, n_chunks)
    y_chunks = np.array_split(y, n_chunks)

    train_performance = np.zeros((n_sigma, n_chunks))
    val_performance = np.zeros((n_sigma, n_chunks))
    is_monotonic = np.ones(n_sigma, dtype=bool)

    nearest_idx_chunks = np.array_split(nearest_idx, n_chunks)

    for i, sigma in enumerate(sigmas):

        for k in range(n_chunks):

            train_chunks = np.setdiff1d(np.arange(n_chunks), [k])
            x_train = np.hstack([x_chunks[j] for j in train_chunks])

            nearest_idx_train = np.hstack([nearest_idx_chunks[j] for j in train_chunks])

            y_train = np.hstack([y_chunks[j] for j in train_chunks])
            y_val = y_chunks[k]

            latent_to_hd = mapping_latent_to_hd(x_train, y_train, latent_state_grid, sigma)

            offset = calc_offset(latent_to_hd, nearest_idx_train, y_train)
            y_pred = apply_reparametrization(latent_to_hd, nearest_idx, offset=offset)

            y_pred_chunks = np.array_split(y_pred, n_chunks)

            y_pred_train = np.hstack([y_pred_chunks[j] for j in train_chunks])
            y_pred_val = y_pred_chunks[k]

            train_performance[i,k] = np.nanmedian(np.abs(np.angle(np.exp(1j*(y_train - y_pred_train)))))
            val_performance[i,k] = np.nanmedian(np.abs(np.angle(np.exp(1j*(y_val - y_pred_val)))))

            dy = np.angle(np.exp(1j * np.diff(latent_to_hd)))
            if np.any(invert * dy < 0):
                is_monotonic[i] = False

    return sigmas, train_performance, val_performance, is_monotonic


def calc_circ_corr(t, x, y):
    """
    Calculate the circular correlation coefficient between two circular variables 
    over time, averaged over time bins.

    Parameters:
    t : array-like
        The time variable.
    x : array-like
        The first circular variable (angles in radians).
    y : array-like
        The second circular variable (angles in radians).

    Returns:
    float
        The average circular correlation coefficient over the time bins.
    """

    t_tmp = t.copy() - t[0]
    delta_t = 30
    bins = int(np.ceil(t_tmp[-1] / delta_t))
    start_inds = [np.where(t_tmp >= delta_t * k)[0][0] for k in range(bins)]
    corr = np.zeros(bins)

    for k in range(bins):
        if k == bins - 1:
            a = x[start_inds[k]:]
            b = y[start_inds[k]:]
        else:
            a = x[start_inds[k]:start_inds[k+1]]
            b = y[start_inds[k]:start_inds[k+1]]

        corr[k] = circ_corr(a, b)

    return np.mean(corr)


def trim_data(t, deconvolved_act, calcium_act, hd, stage, t_start, t_stop, buffer=5):
    """
    Trim the data arrays to a specified time window, with an optional buffer.

    Parameters:
    t : array-like
        The time variable.
    deconvolved_act : array-like
        The deconvolved activity data.
    calcium_act : array-like
        The calcium activity data.
    hd : array-like
        The head direction data.
    stage : array-like
        The stage data.
    t_start : float
        The start time of the window.
    t_stop : float
        The stop time of the window.
    buffer : float, optional
        The buffer time to extend the window on both sides (default is 5).

    Returns:
    tuple
        Trimmed arrays of (t, deconvolved_act, calcium_act, hd, stage).
    """

    t_start = t_start - buffer
    t_stop = t_stop + buffer
    start_ind = np.where(t > t_start)[0][0]
    drop_inds = np.where(t > t_stop)[0]
    if drop_inds.size > 0:
        end_ind = drop_inds[0]
        t = t[start_ind:end_ind]
        deconvolved_act = deconvolved_act[start_ind:end_ind,:]
        calcium_act = calcium_act[start_ind:end_ind,:]
        hd = hd[start_ind:end_ind]
        stage = stage[start_ind:end_ind]
    else:
        deconvolved_act = deconvolved_act[start_ind:,:]
        calcium_act = calcium_act[start_ind:,:]
        hd = hd[start_ind:]
        stage = stage[start_ind:]
        t = t[start_ind:]
    return t, deconvolved_act, calcium_act, hd, stage


def transform_to_uniform(deconvolved_act):
    """
    Perform a quantile transformation on the deconvolved activity data to obtain a uniform distribution,
    while preserving the zero entries.

    Parameters:
    deconvolved_act : array-like
        The deconvolved activity data.

    Returns:
    array-like
        The transformed activity data with a uniform distribution, maintaining zero entries.
    """

    X = deconvolved_act.copy()
    X[X==0] = np.nan
    X = quantile_transform(X, output_distribution='uniform', n_quantiles=X.shape[0])
    X[np.isnan(X)] = 0
    return X


def comp_affinity(deconvolved_act, eps=1):
    """
    Compute the affinity matrix from deconvolved activity data using a Gaussian kernel.

    Parameters:
    deconvolved_act : array-like
        The deconvolved activity data.
    eps : float, optional
        The scale parameter for the Gaussian kernel (default is 1).

    Returns:
    array-like
        The affinity matrix.
    """

    X = transform_to_uniform(deconvolved_act)
    dist = pairwise_distances(X, metric='correlation')
    K = np.exp(-dist**2/eps)
    D = np.sum(K, axis=0)[:,np.newaxis]
    alpha = 1
    K = K / (D * D.T)**alpha
    return K


def compute_embedding(X, eps=0.03, affinity='rbf'):
    """
    Compute the spectral embedding of the input data.

    Parameters:
    X : array-like
        The input data.
    eps : float, optional
        The scale parameter for computing the affinity matrix if the affinity is not precomputed (default is 0.03).
    affinity : str, optional
        The affinity type to be used for spectral embedding. If 'precomputed', X is assumed to be an affinity matrix (default is 'rbf').

    Returns:
    array-like
        The computed spectral embedding with 2 components.
    """

    if not affinity == 'precomputed':
        X = comp_affinity(X, eps=eps)

    X = X.astype('float32')

    embedding = SpectralEmbedding(n_components=2, n_jobs=-1, affinity='precomputed').fit_transform(X)

    return embedding


def plot_embedding(embedding, fig_path, fig_name='embedding', curve=None):
    factor = 1.25
    xmin = factor * np.percentile(embedding[:,0], 1)
    xmax = factor * np.percentile(embedding[:,0], 99)
    ymin = factor * np.percentile(embedding[:,1], 1)
    ymax = factor * np.percentile(embedding[:,1], 99)
    axmin = np.minimum(xmin, ymin)
    axmax = np.maximum(xmax, ymax)

    fig, ax = plt.subplots(figsize=(2, 2))
    im = ax.scatter(embedding[:,0], embedding[:,1], s=0.5, alpha=0.5);

    if curve is not None:
        ax.plot(curve[0], curve[1], 'k--', linewidth=2)

    ax.set_xlim([axmin, axmax])
    ax.set_ylim([axmin, axmax])
    ax.set_aspect('equal');
    ax.set_xlabel('Embedding dimension 1')
    ax.set_ylabel('Embedding dimension 2')
    fig.savefig(os.path.join(fig_path, '{}.pdf'.format(fig_name)), bbox_inches='tight')
    plt.close()
    return axmin, axmax


def plot_embedding_colored(embedding, hd, axmin, axmax, stage_id, fig_path, axis_off=True, high=0.04, vmin=-np.pi, vmax=np.pi):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    im = ax.scatter(embedding[:,0], embedding[:,1], s=0.75, c=hd, vmin=vmin, vmax=vmax, cmap=cmocean.cm.phase, alpha=0.5, plotnonfinite=True)
    ax.set_xlim([axmin, axmax])
    ax.set_ylim([axmin, axmax])
    ax.set_aspect('equal')
    if axis_off:
        plt.axis('off')
    else:
        ax.tick_params(axis='both', which='both', bottom=False, left=False, top=False, labelleft=False, labelbottom=False)
        #ax.set_xlabel('Embedding dimension 1')
        #ax.set_ylabel('Embedding dimension 2')

        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)

        ax.spines['left'].set_bounds(low=0, high=high)
        ax.spines['bottom'].set_bounds(low=0, high=high)

        # Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
        # case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
        # respectively) and the other one (1) is an axes coordinate (i.e., at the very
        # right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
        # actually spills out of the axes.
        ax.plot(-axmin / (-axmin + axmax) + high / (-axmin + axmax), 0, ">k", markersize=3, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, -axmin / (-axmin + axmax) + high / (-axmin + axmax), "^k", markersize=3, transform=ax.get_xaxis_transform(), clip_on=False)

    fig.savefig(os.path.join(fig_path, 'embedding_color=hd_stage={}.pdf'.format(stage_id)), bbox_inches='tight')
    plt.close()


def plot_cv_reparam(sigmas, train_performance, val_performance, is_monotonic, mae_nomap, fig_path):
    plt.figure(figsize=(5,5))
    plt.plot(sigmas * 180 / np.pi, 180 / np.pi * np.mean(train_performance, axis=1), label='training')
    plt.plot(sigmas * 180 / np.pi, 180 / np.pi *  np.mean(val_performance, axis=1), label='validation')
    plt.scatter(sigmas * 180 / np.pi, 180 / np.pi *  np.mean(train_performance, axis=1), c=is_monotonic)
    plt.scatter(sigmas * 180 / np.pi, 180 / np.pi *  np.mean(val_performance, axis=1), c=is_monotonic)
    plt.legend()
    plt.xlabel(r'$\sigma$ [deg]', fontsize=11)
    plt.ylabel('Median absolute error [deg]', fontsize=11)
    plt.xlim(sigmas[0] * 180 / np.pi, sigmas[-1] * 180 / np.pi)
    plt.title('MAE without mapping: {:3.2f} deg'.format(mae_nomap * 180 / np.pi))
    plt.savefig(os.path.join(fig_path, 'cross_validate_reparametrization.pdf'), bbox_inches='tight')
    plt.close()


def plot_mapping_latent_to_hd(latent_state_grid, latent_to_hd, fig_path):
    plt.figure(figsize=(5,5))
    y = latent_to_hd.copy()
    y[np.abs(np.diff(latent_to_hd, append=latent_to_hd[0])) > np.pi] = np.nan
    plt.plot(latent_state_grid, latent_to_hd)
    plt.xlabel('Latent state [rad]', fontsize=11)
    plt.ylabel('Head direction [rad]', fontsize=11)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']);
    plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']);
    plt.savefig(os.path.join(fig_path, 'mapping_latent_to_hd.pdf'), bbox_inches='tight')
    plt.close()


def exclude_data(t, deconvolved_act, calcium_act, hd, stage, exclude_stages):
    """
    Exclude specific stages and zero-activity periods from the data arrays.

    Parameters:
    t : array-like
        The time variable.
    deconvolved_act : array-like
        The deconvolved activity data.
    calcium_act : array-like
        The calcium activity data.
    hd : array-like
        The head direction data.
    stage : array-like
        The stage data.
    exclude_stages : list or array-like
        The stages to be excluded from the data.

    Returns:
    tuple
        Filtered arrays of (t, deconvolved_act, calcium_act, hd, stage) after excluding specified stages and zero-activity periods.
    """

    mask = np.ones(t.size, dtype='bool')
    for k in exclude_stages:
        mask[stage == k] = False
    t = t[mask]
    deconvolved_act = deconvolved_act[mask,:]
    calcium_act = calcium_act[mask,:]
    hd = hd[mask]
    stage = stage[mask]

    mask = np.all(deconvolved_act==0, axis=-1)
    if np.sum(mask) > 0:
        t = t[~mask]
        deconvolved_act = deconvolved_act[~mask,:]
        calcium_act = calcium_act[~mask,:]
        hd = hd[~mask]
        stage = stage[~mask]

    return t, deconvolved_act, calcium_act, hd, stage


def calibration(latent_state, hd, mask, fig_path):
    """
    Calibrate the mapping from latent state to head direction using cross-validation 
    and reparametrization, and generate corresponding plots.

    Parameters:
    latent_state : array-like
        The latent state data.
    hd : array-like
        The head direction data.
    mask : array-like
        A boolean mask to select the data for calibration.
    fig_path : str
        The path to save the generated figures.

    Returns:
    array-like
        The decoded head directions after calibration.
    """

    n_grid = 1000
    latent_state_grid = np.linspace(-np.pi, np.pi, n_grid)
    nearest_idx = np.argmin(np.abs(latent_state[:,np.newaxis] - latent_state_grid[np.newaxis,:]), axis=1)

    rho = calc_circ_corr(t[mask], latent_state[mask], hd[mask])
    invert = 1
    if rho < 0:
        invert = -1

    sigmas, train_performance, val_performance, is_monotonic = cross_validate_reparametrization(latent_state[mask], hd[mask], latent_state_grid, nearest_idx[mask], invert=invert)
    opt_ind = np.argmin(np.mean(val_performance[is_monotonic,:], axis=1))
    sigma_opt = sigmas[is_monotonic][opt_ind]
    latent_to_hd = mapping_latent_to_hd(latent_state[mask], hd[mask], latent_state_grid, sigma_opt)
    offset = calc_offset(latent_to_hd, nearest_idx[mask], hd[mask])
    decoded_direction = apply_reparametrization(latent_to_hd, nearest_idx, offset=offset)

    shift = np.angle(np.nansum(np.exp(1j*(invert*latent_state[mask] - hd[mask]))))
    mae_nomap = np.nanmedian(np.abs(np.angle(np.exp(1j*(invert*latent_state[mask] - shift - hd[mask])))))

    plot_cv_reparam(sigmas, train_performance, val_performance, is_monotonic, mae_nomap, fig_path)
    plot_mapping_latent_to_hd(latent_state_grid, latent_to_hd, fig_path)

    return decoded_direction


def orientation(t, latent_state, hd):
    """
    Determine the orientation of the latent state with respect to head direction
    and compute the circular correlation coefficient.

    Parameters:
    t : array-like
        The time variable.
    latent_state : array-like
        The latent state data.
    hd : array-like
        The head direction data.

    Returns:
    tuple
        The adjusted latent state and the circular correlation coefficient (rho).
    """

    rho = calc_circ_corr(t, latent_state, hd)
    invert = 1
    if rho < 0:
        invert = -1
    return invert * latent_state, rho


def plot_drift_hist(delta, stage_id, fig_path, n_bins=120):
    bins = np.linspace(-np.pi, np.pi, n_bins)
    plt.figure(figsize=(2.0,1.5))
    plt.hist(delta, bins=bins);
    plt.xlabel('Drift [rad]')
    plt.ylabel('Frequency')
    plt.xlim(-np.pi, np.pi)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']);
    plt.savefig(os.path.join(fig_path, 'drift_hist_stage={}.pdf'.format(stage_id)), bbox_inches='tight')
    plt.close()


def plot_drift_over_time(t, delta, stage_id, fig_path):
    filtered_cos = ndimage.median_filter(np.cos(delta), size=80)
    filtered_sin = ndimage.median_filter(np.sin(delta), size=80)
    filtered = np.arctan2(filtered_sin, filtered_cos)
    filtered[np.hstack((np.abs(np.diff(filtered)) > np.pi, False))] = np.nan

    plt.figure(figsize=(8,2))
    plt.scatter(t, delta, s=0.5)
    plt.plot(t, filtered, 'k')
    plt.xlim(t[0], t[-1])
    plt.xticks(300 * np.arange(np.floor(t[-1] / 300 + 1)), 5 * np.arange(np.floor(t[-1] / 300) + 1).astype('int'))
    plt.hlines(0, t[0], t[-1], 'r', alpha=0.5)
    plt.xlabel('Time [min]')
    plt.ylabel('Drift [rad]')
    plt.ylim(-np.pi, np.pi)
    plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']);
    plt.savefig(os.path.join(fig_path, 'drift_over_time_stage={}.pdf'.format(stage_id)), bbox_inches='tight')
    plt.close()


def dump_data(data, recording):
    data_path = os.path.join(BASE_PATH, 'data', 'embedding_data')
    os.makedirs(data_path, exist_ok=True)
    filehandler = open(os.path.join(data_path, '{}.p'.format(recording)), 'wb')
    with filehandler as file:
        pickle.dump(data, file)


def fit_curve(phi, r, eff_num_neighbors=None):
    """
    Fit a curve to the given polar coordinates using a Gaussian kernel 
    and a specified number of effective neighbors.

    Parameters:
    phi : array-like
        The angular coordinates (angles in radians).
    r : array-like
        The radial coordinates.
    eff_num_neighbors : float, optional
        The effective number of neighbors for weighting (default is None, which sets it to 1% of the number of samples).

    Returns:
    tuple
        A tuple containing:
        - phi_curve : array-like
            The angular coordinates of the fitted curve.
        - r_curve : array-like
            The radial coordinates of the fitted curve.
        - x_curve : array-like
            The x-coordinates of the fitted curve in Cartesian coordinates.
        - y_curve : array-like
            The y-coordinates of the fitted curve in Cartesian coordinates.
    """

    if eff_num_neighbors is None:
        n_samples = len(phi)
        eff_num_neighbors = n_samples / 100

    n_curve = 1000
    r_curve = np.zeros(n_curve)
    phi_curve = np.linspace(-np.pi, np.pi, n_curve)
    sigma0 = 0.05

    for i in range(n_curve):
        delta = phi - phi_curve[i]
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        weights = np.exp(-delta**2/sigma0**2)
        r_curve[i] = weighted.median(r, weights)

    x_curve, y_curve = pol2cart(r_curve, phi_curve)

    return phi_curve, r_curve, x_curve, y_curve


def KL(P,Q):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions.

    Parameters:
    P : array-like
        The first probability distribution.
    Q : array-like
        The second probability distribution.

    Returns:
    float
        The KL divergence between the two distributions.
    """
    epsilon = 1e-5
    P = P + epsilon
    Q = Q + epsilon
    divergence = np.sum(P*np.log(P/Q))
    return divergence


def process_eps(eps, deconvolved_act, fig_path):
    """
    Process the given epsilon value to compute the embedding, fit a curve, 
    and evaluate the validity of the embedding based on KL divergence 
    and convex hull checks.

    Parameters:
    eps : float
        The epsilon value used for computing the embedding.
    deconvolved_act : array-like
        The deconvolved activity data.
    fig_path : str
        The path to save the generated figures.

    Returns:
    float
        The median absolute deviation (MAD) of the radial distances if the 
        embedding is valid, otherwise NaN.
    """
    embedding = compute_embedding(deconvolved_act, eps=eps)

    r, phi = cart2pol(embedding[:,0], embedding[:,1])

    valid = True
    n_bins = 36
    phi_bins = np.linspace(-np.pi, np.pi, n_bins+1)
    threshold = 0.4
    ideal_dist = np.ones(n_bins) / n_bins
    empirical_dist = np.histogram(phi, bins=phi_bins, density=False)[0]
    n_samples = embedding.shape[0]
    empirical_dist = empirical_dist / n_samples

    fig_name = 'embedding_eps={:.3f}'.format(eps)
    if KL(empirical_dist, ideal_dist) > threshold:
        valid = False
        axmin, axmax = plot_embedding(embedding, fig_path, fig_name=fig_name, curve=None)
    else:
        phi_curve, r_curve, x_curve, y_curve = fit_curve(phi, r)

        points = np.vstack((x_curve, y_curve)).T
        hull = Delaunay(points)
        in_hull = hull.find_simplex([0,0]) >= 0
        if not in_hull:
            valid = False

        axmin, axmax = plot_embedding(embedding, fig_path, fig_name=fig_name, curve=[x_curve, y_curve])

        n_samples = len(phi)
        nearest_idx = np.zeros(n_samples).astype('int')
        for i in range(n_samples):
            nearest_idx[i] = np.argmin(comp_delta(phi_curve, phi[i])**2)

    if valid:
        mad = np.median(np.abs(r/r_curve[nearest_idx] - 1))
    else:
        mad = np.nan

    return mad


def wait_for_tasks(tasks):
    while True:
        time.sleep(10)
        _, unready = ray.wait(tasks, num_returns=len(tasks), timeout=0)
        if len(unready) == 0:
            break


def find_opt_eps(deconvolved_act, fig_path):
    """
    Find the optimal epsilon value for embedding by evaluating the median absolute 
    deviation (MAD) over a range of epsilon values.

    Parameters:
    deconvolved_act : array-like
        The deconvolved activity data.
    fig_path : str
        The path to save the generated figures and results.

    Returns:
    eps : array-like
        The range of epsilon values tested.
    mad : array-like
        The MAD values corresponding to each epsilon.
    optimal_eps : float
        The epsilon value that minimizes the MAD.
    """

    eps = np.linspace(0.01,0.2,39)

    fig_path = os.path.join(fig_path, 'optimal_eps')
    os.makedirs(fig_path, exist_ok=True)

    mad = [process_eps(e, deconvolved_act, fig_path) for i, e in enumerate(eps)]

    plt.figure(figsize=(3,2))
    plt.plot(eps, mad)
    plt.xlim(0, 0.2)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'Median $|r_i / r_{fit}(x_i, y_i) - 1|$')
    plt.savefig(os.path.join(fig_path, 'mad_vs_eps.pdf'), bbox_inches='tight')

    return eps, mad, eps[np.nanargmin(mad)]


def preprocess(data, smooth, trim, exclude_stages):
    """
    Preprocess the experimental data by optionally smoothing, excluding certain stages, 
    and trimming based on specific time intervals.

    Parameters:
    data : dict
        A dictionary containing the experimental data with keys 'deconv_traces', 'calcium_traces',
        'deconv_sn', 'calcium_sn', 'head_direction', 'stage', and 'time'.
    smooth : bool
        If True, apply a Gaussian filter to smooth the deconvolved activity data.
    trim : bool
        If True, trim the data based on the start and stop times specified in 'stimulus_start' 
        and 'stimulus_stop' in the data dictionary.

    Returns:
    t : array-like
        The preprocessed time data.
    deconvolved_act : array-like
        The preprocessed deconvolved activity data.
    calcium_act : array-like
        The preprocessed calcium activity data.
    deconv_sn : array-like
        The deconvolved signal-to-noise ratios.
    calcium_sn : array-like
        The calcium signal-to-noise ratios.
    hd : array-like
        The preprocessed head direction data.
    stage : array-like
        The preprocessed stage data.
    """
    
    deconvolved_act = data['deconv_traces'].T
    calcium_act = data['calcium_traces'].T
    deconv_sn = data['deconv_sn']
    calcium_sn = data['calcium_sn']
    hd = data['head_direction']
    stage = data['stage'].astype(int)
    t = data['time']

    if len(hd) == 0:
        hd = np.nan * np.zeros_like(t)

    if smooth:
        deconvolved_act = gaussian_filter1d(deconvolved_act, 3, truncate=1, axis=0)

    t, deconvolved_act, calcium_act, hd, stage = exclude_data(t, deconvolved_act, calcium_act, hd, stage, exclude_stages)

    if trim:
        t_start = data['stimulus_start'][0]
        t_stop = data['stimulus_stop'][-1]
        t, deconvolved_act, calcium_act, hd, stage = trim_data(t, deconvolved_act, calcium_act, hd, stage, t_start, t_stop)

    mask = np.logical_not(np.all(np.isnan(deconvolved_act==0), axis=1))
    deconvolved_act = deconvolved_act[mask,:]
    hd = hd[mask]
    stage = stage[mask]
    t = t[mask]

    return t, deconvolved_act, calcium_act, deconv_sn, calcium_sn, hd, stage


def align_decoded_direction_with_hd(decoded_direction, delta, mask):
    """
    Align the decoded direction with the head direction by adjusting for the mean delta.

    Parameters:
    decoded_direction : array-like
        The decoded head direction data.
    delta : array-like
        The difference between the decoded and actual head direction.
    mask : array-like
        A boolean mask to select the relevant data points for alignment.

    Returns:
    decoded_direction : array-like
        The aligned decoded head direction.
    delta : array-like
        The adjusted delta after alignment.
    """
    mean_delta = np.angle(np.nanmean(np.exp(1j * delta[mask])))
    decoded_direction -= mean_delta
    decoded_direction = np.angle(np.exp(1j * decoded_direction))
    delta -= mean_delta
    delta = np.angle(np.exp(1j * delta))
    return decoded_direction, delta


def main(recording, path='.', eps=0.03, smooth=False, exclude_stages=[], reference_stage=0, calibrate=False, orient=False, trim=True, opt=True, align=False):

    with open(os.path.join(path, '{}.p'.format(recording)), 'rb') as file:
        data = pickle.load(file)

    t, deconvolved_act, calcium_act, deconv_sn, calcium_sn, hd, stage = preprocess(data, smooth, trim, exclude_stages)

    fig_path = os.path.join(BASE_PATH, 'figures', recording, 'embedding_figures')
    os.makedirs(fig_path, exist_ok=True)

    if opt:
        eps_list, mad_per_eps, eps = find_opt_eps(deconvolved_act, fig_path)
    else:
        eps_list = []
        mad_per_eps = []

    embedding = compute_embedding(deconvolved_act, eps=eps)

    axmin, axmax = plot_embedding(embedding, fig_path)

    _, latent_state = cart2pol(embedding[:,0], embedding[:,1])    

    rho = None
    if calibrate:
        decoded_direction = calibration(latent_state, hd, stage==reference_stage, fig_path)
    elif orient:
        decoded_direction, rho = orientation(t, latent_state, hd)
    else:
        decoded_direction = latent_state

    if len(hd) > 0:
        delta = comp_delta(decoded_direction, hd)
    else:
        delta = np.nan * np.ones_like(decoded_direction)

    if align:
        decoded_direction, delta = align_decoded_direction_with_hd(decoded_direction, delta, stage==reference_stage)

    hd_tracking = len(hd) == len(decoded_direction) and not np.all(np.isnan(hd))

    stage_list = np.unique(stage)
    for stage_id in stage_list:
        stage_mask = stage == stage_id
        t_stage = t[stage_mask]
        t_stage = t_stage - t_stage[0]

        if hd_tracking:
            plot_embedding_colored(embedding[stage_mask], hd[stage_mask], axmin, axmax, stage_id, fig_path)
            plot_drift_hist(delta[stage_mask], stage_id, fig_path)
            plot_drift_over_time(t_stage, delta[stage_mask], stage_id, fig_path)
            fname = 'comparison_true_vs_decoded_stage={}.pdf'.format(stage_id)
        else:
            fname = '{}_stage={}.pdf'.format(DECODED_DIRECTION, stage_id)
        plot_traces(t_stage, hd[stage_mask], decoded_direction[stage_mask], path=os.path.join(fig_path, fname))

        latent_tuning_curves, bin_centers = comp_tuning_curves(decoded_direction[stage_mask], deconvolved_act[stage_mask,:], n_bins=36)
        latent_tuning_curves = normalize_tuning(latent_tuning_curves)
        if hd_tracking:
            hd_tuning_curves, _ = comp_tuning_curves(hd[stage_mask], deconvolved_act[stage_mask,:], n_bins=36)
            hd_tuning_curves = normalize_tuning(hd_tuning_curves)
            compare_tuning_curves(bin_centers, latent_tuning_curves, hd_tuning=hd_tuning_curves, path=os.path.join(fig_path, 'tuning_curves_stage={}.pdf'.format(stage_id)))
        else:
            compare_tuning_curves(bin_centers, latent_tuning_curves, hd_tuning=None, path=os.path.join(fig_path, 'tuning_curves_stage={}.pdf'.format(stage_id)))
        
        ordering = tuning_heatmap(latent_tuning_curves, label=DECODED_DIRECTION, path=os.path.join(fig_path, 'latent_tuning_heatmap_stage={}.pdf'.format(stage_id)))
        if hd_tracking:
            tuning_heatmap(hd_tuning_curves, ordering=ordering, label='Head direction', path=os.path.join(fig_path, 'hd_tuning_heatmap_stage={}.pdf'.format(stage_id)))

    if 'eye_pos' in data.keys():
        eye_pos = data['eye_pos']
    else:
        eye_pos = np.array([])

    data = {'deconvolved_act': deconvolved_act, 'calcium_act': calcium_act, 'embedding': embedding, 'latent_state': latent_state, DECODED_DIRECTION: decoded_direction,
            'time': t, 'hd': hd, 'stage': stage, 'eps': eps, 'deconv_sn': deconv_sn, 'calcium_sn': calcium_sn, 'eye_pos': eye_pos,
            'stimulus_start': data['stimulus_start'], 'stimulus_stop': data['stimulus_stop'], 'circ_corr': rho, 'eps_list': eps_list, 'mad_per_eps': mad_per_eps,
            'smooth': smooth, 'exclude_stages': exclude_stages, 'reference_stage': reference_stage, 'calibrate': calibrate, 'orient': orient, 'trim': trim,
            'opt': opt, 'align': align}

    dump_data(data, recording)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='spectral embedding')
    parser.add_argument('-r', '--recording', required=True)
    parser.add_argument('-e', '--eps', type=float, default=0.05)
    parser.add_argument('-R', '--reference', type=int, default=0)
    parser.add_argument('-x', '--exclude', type=int, nargs='+')
    parser.add_argument('-s', '--smooth', dest='smooth', action='store_true')
    parser.add_argument('-ns', '--no-smooth', dest='smooth', action='store_false')
    parser.add_argument('-c', '--calibrate', dest='calibrate', action='store_true')
    parser.add_argument('-nc', '--no-calibrate', dest='calibrate', action='store_false')
    parser.add_argument('-t', '--trim', dest='trim', action='store_true')
    parser.add_argument('-nt', '--no-trim', dest='trim', action='store_false')
    parser.add_argument('-o', '--opt', dest='opt', action='store_true')
    parser.add_argument('-no', '--no-opt', dest='opt', action='store_false')
    parser.add_argument('-O', '--orient', dest='orient', action='store_true')
    parser.add_argument('-nO', '--no-orient', dest='orient', action='store_false')
    parser.add_argument('-a', '--align', dest='align', action='store_true')
    parser.add_argument('-na', '--no-align', dest='align', action='store_false')
    parser.add_argument('-p', '--path', default=os.path.join(BASE_PATH, 'data', 'base_data'))
    parser.set_defaults(smooth=True)
    parser.set_defaults(calibrate=False)
    parser.set_defaults(trim=False)
    parser.set_defaults(orient=False)
    parser.set_defaults(align=False)
    parser.set_defaults(opt=True)

    args = parser.parse_args()

    if args.exclude is None:
        exclude_stages = []
    else: 
        exclude_stages = args.exclude

    os.makedirs(os.path.join(BASE_PATH, 'data', 'embedding_data'), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, 'figures'), exist_ok=True)

    main(args.recording, path=args.path, eps=args.eps, smooth=args.smooth, exclude_stages=exclude_stages, reference_stage=args.reference, calibrate=args.calibrate, orient=args.orient, trim=args.trim, align=args.align, opt=args.opt)

