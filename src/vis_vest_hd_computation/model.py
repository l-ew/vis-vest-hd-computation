from scipy.optimize import minimize
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
from vis_vest_hd_computation.utils import get_temp_df
from vis_vest_hd_computation.imports import *


def stim_vel(t, omega, vmax):
    """
    Calculate the stimulus velocity at time `t`.

    Parameters:
    t : scalar or array-like
        The time or array of time points at which to calculate the stimulus velocity.
    omega : float
        Two times the angular frequency of the sinusoidal function.
    vmax : float
        The maximum velocity of the stimulus.

    Returns:
    float or numpy.ndarray
        The calculated velocity at time `t`, or an array of velocities if `t` is array-like.
    """
    if np.isscalar(t):
        if t < np.pi / omega:
            return vmax * np.sin(omega * t)
        else:
            return 0
    else:
        out = vmax * np.sin(omega * t)
        out[t > np.pi / omega] = 0
        return out


def retinal_slip_transform(x, alpha, beta):
    """
    Transform the retinal slip velocity using a nonlinear function.

    This function applies a nonlinear transformation to the retinal slip velocity `x`. 
    The transformation is defined by a power-law with exponent `alpha` and an exponential 
    decay with rate `beta`.

    Parameters:
    x : float or array-like
        The retinal slip velocity or an array of slip velocities to be transformed.
    alpha : float
        The exponent of the power-law transformation.
    beta : float
        The rate of the exponential decay.

    Returns:
    float or numpy.ndarray
        The transformed retinal slip velocity, or an array of transformed velocities if `x` is array-like.
    """
    z = np.sign(x) * np.abs(x)**alpha * np.exp(-beta * np.abs(x))
    return z


def source(t, vmax, omega):
    """
    Calculate the source signal at time `t`.

    This function computes the source signal based on a cosine function. 
    If the time `t` is a scalar and less than π/omega, the function returns `omega * vmax * cos(omega * t)`. 
    Otherwise, it returns 0. If `t` is an array, the function applies the same logic element-wise.

    Parameters:
    t : scalar or array-like
        The time or array of time points at which to calculate the source signal.
    vmax : float
        The maximum velocity of the source signal.
    omega : float
        Two times the angular frequency of the cosine function.

    Returns:
    float or numpy.ndarray
        The calculated source signal at time `t`, or an array of source signals if `t` is array-like.
    """
    if np.isscalar(t):
        if t < np.pi / omega:
            return omega * vmax * np.cos(omega * t)
        else:
            return 0
    else:
        out = omega * vmax * np.cos(omega * t)
        out[t > np.pi / omega] = 0
        return out


def dc(c, acc, tau=TAU_C):
    """
    Compute the derivative canal signal given the canal signal and the angular acceleration.

    Parameters:
    c : float or array-like
        Canal signal.
    acc : float or array-like
        Angular acceleration.
    tau : float, optional
        Canal time constant. Default is `TAU_C`.

    Returns:
    float or numpy.ndarray
        Derivative canal signal.
    """
    return -c / tau + acc


def dy(y, t, tau, vmax, omega):
    """
    Compute the derivative canal signal given the time, the canal time constant, the maximum velocity, and the variable omega related to the angular frequency.

    Parameters:
    y : array-like
        Canal signal.
    t : float
        The current time point.
    tau : float
        Canal time constant.
    vmax : float
        The maximum velocity for the source function.
    omega : float
        The angular frequency for the source function.

    Returns:
    list
        Derivative canal signal.
    """
    acc = source(t, vmax, omega)
    dydt = [dc(y[0], acc, tau=tau)]
    return dydt


def vestibular_signal(t_interp, vmax, T=T_TRANSIENT_STIM, tau=TAU_C):
    """
    Generate the canal signal and its derivative for a given set of parameters.

    Parameters:
    t_interp : array-like
        The time points at which to interpolate the canal signal.
    vmax : float
        The maximum velocity for the sinusoidal input.
    T : float, optional
        Half period of the sinusoidal input. Default is 3.
    tau : float, optional
        The canal time constant. Default is `TAU_C`.

    Returns:
    tuple of numpy.ndarray
        c : numpy.ndarray
            Canal signal.
        dcdt : numpy.ndarray
            Derivative canal signal.
    """
    omega = np.pi / T
    y0 = [0]
    sol = odeint(dy, y0, t_interp, args=(tau, vmax, omega))
    c = sol[:, 0]
    acc = source(t_interp, vmax, omega)
    dcdt = dc(c, acc, tau=tau)
    return c, dcdt


def surround_vel(t, omega, vmax, static_grating_interval):
    """
    Calculate the surround velocity based on a sinusoidal function.

    Parameters:
    t : numpy.ndarray
        The time points at which to calculate the surround velocity.
    omega : float
        Two times the angular frequency of the sinusoidal function.
    vmax : float
        The maximum velocity of the sinusoidal function.
    static_grating_interval : int
        Determines the interval during which the velocity is set to zero. 
        If 0, velocity is zero for the first half of the period T.
        If 1, velocity is zero for the second half of the period T.

    Returns:
    numpy.ndarray
        The calculated surround velocity at each time point `t`.
    """
    v = vmax * np.sin(omega * t)
    T = np.pi / omega
    if static_grating_interval == 0:
        v[t < T / 2] = 0
    elif static_grating_interval == 1:
        v[t > T / 2] = 0
    v[t > T] = 0
    return v


def comp_inputs(df, t, T, tau, sensory_transduction=True):
    """
    Compute various input signals for the model from a DataFrame.

    This function computes the platform velocity `v_platform`, canal signal `c`, its derivative 
    `dcdt`, and surround velocity `v_surround` based on the input DataFrame `df`. It applies the 
    `stim_vel`, `vestibular_signal`, `source`, and `surround_vel` functions to generate these signals 
    for each row of the DataFrame.

    Parameters:
    df : pandas.DataFrame
        The input DataFrame containing columns 'motor_velocity' and 'grating_velocity', as well as 
        'static_grating_interval'.
    t : numpy.ndarray
        The time points at which to calculate the signals.
    T : float
        Half period of the sinusoidal input.
    tau : float
        The time constant for the decay of the canal signal.
    sensory_transduction : bool, optional
        A flag indicating whether to apply sensory transduction. Default is True.

    Returns:
    tuple of numpy.ndarray
        v_platform : numpy.ndarray
            The platform velocity for each sample.
        c : numpy.ndarray
            The canal signal for each sample.
        dcdt : numpy.ndarray
            The derivative canal signal for each sample.
        v_surround : numpy.ndarray
            The surround velocity for each sample.
    """

    v_platform = df.apply(lambda x: stim_vel(t, np.pi / T, x['motor_velocity'] * np.pi / 180), axis=1)
    v_platform = np.vstack(v_platform)

    if sensory_transduction:
        c_dcdt = df.apply(lambda x: vestibular_signal(t, x['motor_velocity'] * np.pi / 180, T=T, tau=tau), axis=1)
        c = [x[0] for x in c_dcdt]
        dcdt = [x[1] for x in c_dcdt]
        c = np.vstack(c)
        dcdt = np.vstack(dcdt)
    else:
        c = v_platform.copy()
        dcdt = df.apply(lambda x: source(t, np.pi / T, x['motor_velocity'] * np.pi / 180), axis=1)
        dcdt = np.vstack(dcdt)

    v_surround = df.apply(lambda x: surround_vel(t, np.pi / T, x['grating_velocity'] * np.pi / 180, x['static_grating_interval']), axis=1)
    v_surround = np.vstack(v_surround)

    return v_platform, c, dcdt, v_surround


def time_params(T, k=360):
    n_steps = int(k * T + 1)
    dt = 1 / k
    t = np.linspace(0, T, n_steps)
    return k, n_steps, dt, t


def predict_vel(model_params, v_platform, c, dcdt, v_surround, dt, fit_gamma_c=True, fit_gamma_r=True, plant_lag=None, plant_coef=None, dark=False, frmd7=False, simple_model=False):
    """
    Predict the angular head velocity estimate and eye velocity over time using a specified model.

    This function computes the angular head velocity estimate `v_est` and eye velocity `v_eye` 
    of the system over time using a set of model parameters. It also calculates an intermediate 
    variable `z` based on the retinal slip. The calculation considers platform velocity `v_platform`, 
    canal signal `c`, its derivative canal signal `dcdt`, surround velocity `v_surround`, and time step `dt`.
    
    Parameters:
    model_params : list of floats
        The parameters of the model.
    v_platform : numpy.ndarray
        The platform velocity. Shape should be (n_samples, n_steps).
    c : numpy.ndarray
        The canal signal. Shape should be (n_samples, n_steps).
    dcdt : numpy.ndarray
        The derivative canal signal. Shape should be (n_samples, n_steps).
    v_surround : numpy.ndarray
        The surround velocity. Shape should be (n_samples, n_steps).
    dt : float
        The time step for the simulation.
    fit_gamma_c : bool
        Whether to fit a canal dependence of the vestibular gain, or set the slope to zero.
    fit_gamma_r : bool
        Whether to fit a canal dependence of the visual gain, or set the slope to zero.
        Time lag of oculomotor plant.
    plant_coef : numpy.ndarray
        Coefficients of oculomotor plant.
    dark : bool, optional
        A flag indicating whether the simulation is in the dark. Default is False.
    frmd7 : bool, optional
        A flag indicating whether the animal has FRMD7 mutation.
        
    Returns:
    tuple of numpy.ndarray
        v_est : numpy.ndarray
            The angular head velocity estimate. Shape is (n_samples, n_steps).
        v_eye : numpy.ndarray
            The eye velocity. Shape is (n_samples, n_steps).
    """

    n_samples, n_steps = v_platform.shape
    v_est = np.zeros((n_samples, n_steps))
    v_eye = np.zeros((n_samples, n_steps))
    z = np.zeros((n_samples, n_steps))

    if plant_lag is None:
        tau_est, g_r0, g_c0, gamma_r, gamma_c, b1, b2, b3, plant_lag, alpha, beta = model_params
    elif fit_gamma_c and fit_gamma_r:
        tau_est, g_r0, g_c0, gamma_r, gamma_c, alpha, beta = model_params
        a1, a2, a3 = plant_coef
    elif fit_gamma_c:
        tau_est, g_r0, g_c0, gamma_c, alpha, beta = model_params
        a1, a2, a3 = plant_coef
        gamma_r = 0.
    elif fit_gamma_r:
        tau_est, g_r0, g_c0, gamma_r, alpha, beta = model_params
        a1, a2, a3 = plant_coef
        gamma_c = 0.
    else:
        tau_est, g_r0, g_c0, alpha, beta = model_params
        a1, a2, a3 = plant_coef
        gamma_r = 0.
        gamma_c = 0.

    r = np.zeros(n_samples)

    for i in range(n_steps-1):

        if not simple_model:
            motor_command = -(a1 * v_est[:,i] + a2 * c[:,i] + a3 * dcdt[:,i])
            v_eye[:,i+1] = v_eye[:,i] + dt * (motor_command - v_eye[:,i]) / plant_lag
        else:
            v_eye[:,i+1] = -v_est[:,i]

        r = -v_eye[:,i] - v_platform[:,i] + v_surround[:,i]

        if not dark and not frmd7:
            z[:,i] = retinal_slip_transform(r, alpha, beta)

        g_r = g_r0 + gamma_r * np.abs(c[:,i])
        g_c = g_c0 + gamma_c * np.abs(c[:,i])
        g_c[g_c>1] = 1

        v_est[:,i+1] = v_est[:,i] + dt * (-g_r * z[:,i] + g_c * c[:,i] - v_est[:,i]) / tau_est

    return v_est, v_eye


def predict_rotation(model_params, v_platform, c, dcdt, v_surround, dt, fit_gamma_c=True, fit_gamma_r=True, plant_lag=None, plant_coef=None, dark=False, frmd7=False):
    """
    Predict the cumulative rotation based on the angular head velocity estimate.

    This function computes the cumulative rotation by integrating the angular head velocity estimate 
    `v_est` over time. The calculation considers platform velocity `v_platform`, canal signal `c`, 
    its derivative canal signal `dcdt`, surround velocity `v_surround`, and time step `dt`.

    Parameters:
    model_params : list of floats
        The parameters of the model.
    v_platform : numpy.ndarray
        The platform velocity. Shape should be (n_samples, n_steps).
    c : numpy.ndarray
        The canal signal. Shape should be (n_samples, n_steps).
    dcdt : numpy.ndarray
        The derivative canal signal. Shape should be (n_samples, n_steps).
    v_surround : numpy.ndarray
        The surround velocity. Shape should be (n_samples, n_steps).
    dt : float
        The time step for the simulation.
    fit_gamma_c : bool
        Whether to fit a canal dependence of the vestibular gain, or set the slope to zero.
    fit_gamma_r : bool
        Whether to fit a canal dependence of the visual gain, or set the slope to zero.
    plant_lag : float
        Time lag of oculomotor plant.
    plant_coef : numpy.ndarray
        Coefficients of oculomotor plant.
    dark : bool, optional
        A flag indicating whether the simulation is in the dark. Default is False.
    frmd7 : bool, optional
        A flag indicating whether the animal has FRMD7 mutation.
        
    Returns:
    numpy.ndarray
        The cumulative rotation calculated by integrating the angular head velocity estimate. Shape is (n_samples,).
    """
    v_est, v_eye = predict_vel(model_params, v_platform, c, dcdt, v_surround, dt, fit_gamma_c=fit_gamma_c, fit_gamma_r=fit_gamma_r, plant_lag=plant_lag, plant_coef=plant_coef, dark=dark, frmd7=frmd7)
    rot_est = np.sum(v_est * dt, axis=-1)
    neg_eye_rot = -np.sum(v_eye * dt, axis=-1)
    return rot_est, neg_eye_rot


def loss_rotation(model_params, v_platform, c, dcdt, v_surround, internal_rotation, weights, dt, fit_gamma_c=True, fit_gamma_r=True, plant_lag=None, plant_coef=None):
    """
    Calculate the mean squared error loss between the total predicted and the total internal rotation.

    Parameters:
    model_params : list of floats
        The parameters of the model.
    v_platform : numpy.ndarray
        The platform velocity. Shape should be (n_samples, n_steps).
    c : numpy.ndarray
        The canal signal. Shape should be (n_samples, n_steps).
    dcdt : numpy.ndarray
        The derivative canal signal. Shape should be (n_samples, n_steps).
    v_surround : numpy.ndarray
        The surround velocity. Shape should be (n_samples, n_steps).
    internal_rotation : numpy.ndarray
        The actual internal rotation values to compare against. Shape should be (n_samples,).
    dt : float
        The time step for the simulation.
    fit_gamma_c : bool
        Whether to fit a canal dependence of the vestibular gain, or set the slope to zero.
    fit_gamma_r : bool
        Whether to fit a canal dependence of the visual gain, or set the slope to zero.
    plant_lag : float
        Time lag of oculomotor plant.
    plant_coef : numpy.ndarray
        Coefficients of oculomotor plant.
        
    Returns:
    float
        The mean squared error loss between the total predicted and the total internal rotation.
    """
    pred_rotation, _ = predict_rotation(model_params, v_platform, c, dcdt, v_surround, dt, fit_gamma_c=fit_gamma_c, visual_canal_slope=visual_canal_slope, plant_lag=plant_lag, plant_coef=plant_coef)
    loss = mean_squared_error(internal_rotation, pred_rotation, sample_weight=weights)
    return loss


def loss_vel(model_params, v_platform, c, dcdt, v_surround, target_vel, weights, dt, skip, fit_gamma_c=True, fit_gamma_r=True, plant_lag=None, plant_coef=None):
    """
    Calculate the weighted mean squared error loss between predicted and target eye velocities.

    Parameters:
    model_params : list of floats
        The parameters of the model.
    v_platform : numpy.ndarray
        The platform velocity. Shape should be (n_samples, n_steps).
    c : numpy.ndarray
        The canal signal. Shape should be (n_samples, n_steps).
    dcdt : numpy.ndarray
        The derivative canal signal. Shape should be (n_samples, n_steps).
    v_surround : numpy.ndarray
        The surround velocity. Shape should be (n_samples, n_steps).
    target_vel : numpy.ndarray
        The target eye velocity to compare against. Shape should be (n_samples, n_steps).
    weights : numpy.ndarray
        The sample weights for the loss calculation. Shape should be (n_samples,).
    dt : float
        The time step for the simulation.
    skip : int
        The interval at which to sample the time series for the loss calculation.
    fit_gamma_c : bool
        Whether to fit a canal dependence of the vestibular gain, or set the slope to zero.
    fit_gamma_r : bool
        Whether to fit a canal dependence of the visual gain, or set the slope to zero.
    plant_lag : float
        Time lag of oculomotor plant.
    plant_coef : numpy.ndarray
        Coefficients of oculomotor plant.
        
    Returns:
    float
        The weighted mean squared error loss between the predicted and target eye velocities.
    """
    _, v_eye = predict_vel(model_params, v_platform, c, dcdt, v_surround, dt, fit_gamma_c=fit_gamma_c, fit_gamma_r=fit_gamma_r, plant_lag=plant_lag, plant_coef=plant_coef, dark=False)
    loss = mean_squared_error(target_vel, v_eye[:,::skip], sample_weight=weights)
    return loss


def loss_dir(model_params, v_platform, c, dcdt, v_surround, target_dir, weights, dt, skip, fit_gamma_c=True, fit_gamma_r=True, plant_lag=None, plant_coef=None):
    """
    Calculate the weighted mean squared error loss between predicted and target directions.

    Parameters:
    model_params : list of floats
        The parameters of the model.
    v_platform : numpy.ndarray
        The platform velocity. Shape should be (n_samples, n_steps).
    c : numpy.ndarray
        The canal signal. Shape should be (n_samples, n_steps).
    dcdt : numpy.ndarray
        The derivative canal signal. Shape should be (n_samples, n_steps).
    v_surround : numpy.ndarray
        The surround velocity. Shape should be (n_samples, n_steps).
    target_dir : numpy.ndarray
        The target direction to compare against. Shape should be (n_samples, n_steps).
    weights : numpy.ndarray
        The sample weights for the loss calculation. Shape should be (n_samples,).
    dt : float
        The time step for the simulation.
    skip : int
        The interval at which to sample the time series for the loss calculation.
    fit_gamma_c : bool
        Whether to fit a canal dependence of the vestibular gain, or set the slope to zero.
    fit_gamma_r : bool
        Whether to fit a canal dependence of the visual gain, or set the slope to zero.
    plant_lag : float
        Time lag of oculomotor plant.
    plant_coef : numpy.ndarray
        Coefficients of oculomotor plant.

    Returns:
    float
        The weighted mean squared error loss between the predicted and target directions.
    """
    v_est, v_eye = predict_vel(model_params, v_platform, c, dcdt, v_surround, dt, fit_gamma_c=fit_gamma_c, fit_gamma_r=fit_gamma_r, plant_lag=plant_lag, plant_coef=plant_coef, dark=False)
    pred_dir = np.cumsum(v_est * dt, axis=-1)
    loss = mean_squared_error(target_dir, pred_dir[:,::skip], sample_weight=weights)
    return loss


def loss_combined(model_params, v_platform, c, dcdt, v_surround, target_vel, target_dir, sample_weight_vel, sample_weight_dir, dt, skip, dark, fit_gamma_c=True, fit_gamma_r=True, plant_lag=None, plant_coef=None, lmbda=0.0):
    """
    Calculate the combined weighted mean squared error loss for both velocity and direction.

    This function computes a combined loss that includes the weighted mean squared error (MSE) loss 
    for both predicted eye velocity (`v_eye`) and predicted direction (`pred_dir`). The combined loss 
    is weighted by a parameter `lmbda` to balance the contributions of each individual loss.

    Parameters:
    model_params : list of floats
        The parameters of the model.
    v_platform : numpy.ndarray
        The platform velocity. Shape should be (n_samples, n_steps).
    c : numpy.ndarray
        The canal signal. Shape should be (n_samples, n_steps).
    dcdt : numpy.ndarray
        The derivative canal signal. Shape should be (n_samples, n_steps).
    v_surround : numpy.ndarray
        The surround velocity. Shape should be (n_samples, n_steps).
    target_vel : numpy.ndarray
        The target eye velocity to compare against. Shape should be (n_samples, n_steps).
    target_dir : numpy.ndarray
        The target direction to compare against. Shape should be (n_samples, n_steps).
    sample_weight_vel : numpy.ndarray
        The sample weights for the velocity loss calculation. Shape should be (n_samples,).
    sample_weight_dir : numpy.ndarray
        The sample weights for the direction loss calculation. Shape should be (n_samples,).
    dt : float
        The time step for the simulation.
    skip : int
        The interval at which to sample the time series for the loss calculation.
    dark : bool or numpy.ndarray
        A flag or array indicating whether the simulation is in the dark.
    fit_gamma_c : bool
        Whether to fit a canal dependence of the vestibular gain, or set the slope to zero.
    fit_gamma_r : bool
        Whether to fit a canal dependence of the visual gain, or set the slope to zero.
    plant_lag : float
        Time lag of oculomotor plant.
    plant_coef : numpy.ndarray
        Coefficients of oculomotor plant.
    lmbda : float, optional
        The weighting factor for balancing velocity and direction loss. Default is 0.0.

    Returns:
    float
        The combined weighted mean squared error loss.
    """
    if np.any(dark):
        v_est, v_eye = predict_vel(model_params, v_platform[~dark,:], c[~dark,:], dcdt[~dark,:], v_surround[~dark,:], dt, fit_gamma_c=fit_gamma_c, fit_gamma_r=fit_gamma_r, plant_lag=plant_lag, plant_coef=plant_coef, dark=False)
        loss1 = mean_squared_error(target_vel, v_eye[:,::skip], sample_weight=sample_weight_vel)

        v_est, v_eye = predict_vel(model_params, v_platform[dark,:], c[dark,:], dcdt[dark,:], v_surround[dark,:], dt, fit_gamma_c=fit_gamma_c, fit_gamma_r=fit_gamma_r, plant_lag=plant_lag, plant_coef=plant_coef, dark=True)
        pred_dir = np.cumsum(v_est * dt, axis=-1)
        loss2 = mean_squared_error(target_dir, pred_dir[:,::skip], sample_weight=sample_weight_dir)
    else:
        v_est, v_eye = predict_vel(model_params, v_platform, c, dcdt, v_surround, dt, fit_gamma_c=fit_gamma_c, fit_gamma_r=fit_gamma_r, plant_lag=plant_lag, plant_coef=plant_coef, dark=False)
        loss1 = mean_squared_error(target_vel, v_eye[:,::skip], sample_weight=sample_weight_vel)
        pred_dir = np.cumsum(v_est * dt, axis=-1)
        loss2 = mean_squared_error(target_dir, pred_dir[:,::skip], sample_weight=sample_weight_dir)

    combined_loss = (1 - lmbda) * loss1 + lmbda * loss2

    return combined_loss


def model_pred(df, model_params, tau=TAU_C, T=T_TRANSIENT_STIM, fps=FPS_MINISCOPE, fit_gamma_c=True, fit_gamma_r=True, plant_lag=None, plant_coef=None, dark=False, frmd7=False, t_post=0):
    """
    Generate model predictions for angular head velocity and rotation estimates and eye movement.

    Parameters:
    df : pandas.DataFrame
        DataFrame with rows corresponding to different stimuli and columns, an index specifying the stimulus and columns for all time steps.
    model_params : list of floats
        The parameters of the model.
    tau : float, optional
        The time constant for the decay of the canal signal. Default is `TAU_C`.
    T : float, optional
        The period of the transient stimulus. Default is `T_TRANSIENT_STIM`.
    fps : int, optional
        Miniscope frames per second. Default is FPS_MINISCOPE.
    fit_gamma_c : bool
        Whether to fit a canal dependence of the vestibular gain, or set the slope to zero.
    fit_gamma_r : bool
        Whether to fit a canal dependence of the visual gain, or set the slope to zero.
    plant_lag : float
        Time lag of oculomotor plant.
    plant_coef : numpy.ndarray
        Coefficients of oculomotor plant.
    dark : bool, optional
        A flag indicating whether the simulation is in the dark. Default is False.
    frmd7 : bool, optional
        A flag indicating whether the animal has FRMD7 mutation.
    t_post : float, optional
        Additional time after the transient stimulus to include in the simulation. Default is 0.

    Returns:
    tuple
        df : pandas.DataFrame
            The input DataFrame updated with model predictions for the total rotation estimates.
        df_time_series : pandas.DataFrame
            DataFrame containing the time series of the predicted rotation.
        df_eye_vel : pandas.DataFrame
            DataFrame containing the time series of the predicted eye velocity.
        df_eye_pos : pandas.DataFrame
            DataFrame containing the time series of the predicted eye position.
    """
    k, n_steps, dt, t = time_params(T+t_post)
    v_platform, c, dcdt, v_surround = comp_inputs(df, t, T, tau)
    skip = k // fps

    rot_est, neg_eye_rot = predict_rotation(model_params, v_platform, c, dcdt, v_surround, dt, fit_gamma_c=fit_gamma_c, fit_gamma_r=fit_gamma_r, plant_lag=plant_lag, plant_coef=plant_coef, dark=dark, frmd7=frmd7)
    v_est, v_eye = predict_vel(model_params, v_platform, c, dcdt, v_surround, dt, fit_gamma_c=fit_gamma_c, fit_gamma_r=fit_gamma_r, plant_lag=plant_lag, plant_coef=plant_coef, dark=dark, frmd7=frmd7)
    pred_rotation_1st_half = np.sum(v_est[:,:n_steps//2] * dt, axis=-1)

    df['dark'] = dark
    df['animal'] = 'model'
    df[MODEL_ROTATION] = rot_est
    df[MODEL_NEG_EYE_ROTATION] = neg_eye_rot
    df['{}_1st_half'.format(MODEL_ROTATION)] = pred_rotation_1st_half

    n_downsampled = (n_steps - 1) // skip + 1
    rotation = np.cumsum(v_est * dt, axis=-1)
    df_time_series = get_temp_df(df, rotation[:,::skip], T+t_post, n_downsampled, dark=dark, value_name=MODEL_ROTATION)

    df_eye_vel = get_temp_df(df, v_eye[:,::skip], T+t_post, n_downsampled, dark=dark, value_name=EYE_VELOCITY)

    rotation = np.cumsum(-v_eye * dt, axis=-1)
    df_eye_pos = get_temp_df(df, rotation[:,::skip], T+t_post, n_downsampled, dark=dark, value_name=MODEL_NEG_EYE_ROTATION)

    return df, df_time_series, df_eye_vel, df_eye_pos


def comp_sample_weight_factor(static_grating_interval):
    """
    Compute multiplicative factor to sample weights based on static grating intervals.

    Parameters:
    static_grating_interval : numpy.ndarray
        The array indicating the static grating interval for each sample.

    Returns:
    numpy.ndarray
        The computed weight factor for each element in `static_grating_interval`.
    """
    weights = np.zeros(len(static_grating_interval), dtype='float')
    weights[static_grating_interval==-1] = 1
    weights[static_grating_interval==0] = 2.
    weights[static_grating_interval==1] = 2.
    return weights


def comp_sample_weight(x):
    """
    Compute sample weights based on the maximum absolute values in the input array.

    This function calculates sample weights inversely proportional to the square of the maximum 
    absolute values of each sample in the input array `x`. An epsilon value is added to avoid 
    division by zero and to ensure numerical stability.

    Parameters:
    x : numpy.ndarray
        The input array for which to compute sample weights. Shape should be (n_samples, n_steps).

    Returns:
    numpy.ndarray
        The computed sample weights for each sample. Shape is (n_samples,).
    """
    eps = 10 * np.pi / 180
    sample_weight = np.max(np.abs(x), axis=-1) + eps
    sample_weight = 1 / sample_weight**2
    return sample_weight


def init_params(plant_lag, plant_coef, fit_gamma_c, fit_gamma_r):
    tau_est = 0.2
    g_r0 = 1.0
    g_c0 = 0.1
    gamma_r = 1.0
    gamma_c = 0.1
    alpha = 1.0
    beta = 1.0

    if plant_coef is None:
        b1 = 0.94
        b2 = 0.14
        b3 = 0.19
        plant_lag = 0.17
        x0 = [tau_est, g_r0, g_c0, gamma_r, gamma_c, b1, b2, b3, plant_lag, alpha, beta]
    elif fit_gamma_c and fit_gamma_r:
        x0 = [tau_est, g_r0, g_c0, gamma_r, gamma_c, alpha, beta]
    elif fit_gamma_c:
        x0 = [tau_est, g_r0, g_c0, gamma_c, alpha, beta]
    elif fit_gamma_r:
        x0 = [tau_est, g_r0, g_c0, gamma_r, alpha, beta]
    else:
        x0 = [tau_est, g_r0, g_c0, alpha, beta]

    return x0


def fit_model(df, df_dark=None, fps=FPS_MINISCOPE, T=T_TRANSIENT_STIM, tau=TAU_C, x0=None, fit_gamma_r=True, fit_gamma_c=True, plant_lag=None, plant_coef=None, norm=True, loss_type='eye_velocity', lmbda=0.5, t_post=0):

    k, n_steps, dt, t = time_params(T+t_post)

    df_eye = df[['motor_velocity', 'grating_velocity', 'static_grating_interval', 't', EYE_VELOCITY]]

    if df_dark is not None:
        df_dir = df_dark[['motor_velocity', 'grating_velocity', 'static_grating_interval', 't', DECODED_ROTATION]]
    else:
        df_dir = df[['motor_velocity', 'grating_velocity', 'static_grating_interval', 't', DECODED_ROTATION]]

    def process_df(df, var):
        df = df.pivot(index=['motor_velocity', 'grating_velocity', 'static_grating_interval'], columns='t')[var]
        return df.reset_index()

    df_eye = process_df(df_eye, EYE_VELOCITY)
    df_dir = process_df(df_dir, DECODED_ROTATION)

    skip = k // fps
    v_platform, c, dcdt, v_surround = comp_inputs(df_eye, t, T, tau)
    weight_factors = comp_sample_weight_factor(df_eye['static_grating_interval'].to_numpy())

    dark = np.zeros(df_eye.shape[0] + df_dir.shape[0], dtype=bool)
    if df_dark is not None:
        dark[df_eye.shape[0]:] = True
        v_platform_dark, c_dark, dcdt_dark, v_surround_dark = comp_inputs(df_dir, t, T, tau)
        v_platform = np.vstack((v_platform, v_platform_dark))
        c = np.vstack((c, c_dark))
        dcdt = np.vstack((dcdt, dcdt_dark))
        v_surround = np.vstack((v_surround, v_surround_dark))

    target_velocities = df_eye.iloc[:,3:].to_numpy()
    target_directions = df_dir.iloc[:,3:].to_numpy()

    if norm:
        sample_weight_vel = comp_sample_weight(target_velocities) * weight_factors
        sample_weight_dir = comp_sample_weight(target_directions) * weight_factors
    else:
        sample_weight_vel = None
        sample_weight_dir = None

    if loss_type == 'eye_velocity':
        loss_fun = lambda x: loss_vel(x, v_platform, c, dcdt, v_surround, target_velocities, sample_weight_vel, dt, skip, fit_gamma_c=fit_gamma_c, fit_gamma_r=fit_gamma_r, plant_lag=plant_lag, plant_coef=plant_coef)
    elif loss_type == 'decoded_rotation':
        loss_fun = lambda x: loss_dir(x, v_platform, c, dcdt, v_surround, target_directions, sample_weight_dir, dt, skip, fit_gamma_c=fit_gamma_c, fit_gamma_r=fit_gamma_r, plant_lag=plant_lag, plant_coef=plant_coef)
    elif loss_type == 'combined':
        loss_fun = lambda x: loss_combined(x, v_platform, c, dcdt, v_surround, target_velocities, target_directions, sample_weight_vel, sample_weight_dir, dt, skip, dark, fit_gamma_c=fit_gamma_c, fit_gamma_r=fit_gamma_r, plant_lag=plant_lag, plant_coef=plant_coef, lmbda=lmbda)

    if x0 is None:
        x0 = init_params(plant_lag, plant_coef, fit_gamma_c, fit_gamma_r)
    res = minimize(loss_fun, x0, options={'disp': True}, method='BFGS')

    model_params = res.x

    if sample_weight_vel is not None:
        sample_weights_vel = sample_weight_vel[:,np.newaxis] * np.ones_like(target_velocities)
        sample_weights_dir = sample_weight_dir[:,np.newaxis] * np.ones_like(target_directions)
        sample_weights_dir = sample_weights_dir.flatten()
        sample_weights_vel = sample_weights_vel.flatten()

    return model_params, loss_fun, sample_weights_vel, sample_weights_dir


def adjust_vestibular_gain_in_darkness(df, model_params, fit_gamma_c=True, fit_gamma_r=True, plant_lag=None, plant_coef=None, t_post=0):

    gain_vals = np.linspace(0,1,41)
    rmse = []

    from sklearn.metrics import mean_squared_error

    for g in gain_vals:
        p = model_params.copy()
        p[2] = g
        df_est_rotation = model_pred(df[df['motor_velocity'] > 0].copy(), p, fit_gamma_c=fit_gamma_c, fit_gamma_r=fit_gamma_r, plant_lag=plant_lag, plant_coef=plant_coef, dark=True, t_post=t_post)[0]
        rmse.append(mean_squared_error(df_est_rotation[DECODED_ROTATION], df_est_rotation[MODEL_ROTATION], squared=False))

    g = gain_vals[np.argmin(rmse)]
    p = model_params.copy()
    p[2] = g
    df_est_rotation, df_est_rotation_time_series, _, _= model_pred(df, p, fit_gamma_c=fit_gamma_c, fit_gamma_r=fit_gamma_r, plant_lag=plant_lag, plant_coef=plant_coef, dark=True, t_post=t_post)

    return df_est_rotation, df_est_rotation_time_series
