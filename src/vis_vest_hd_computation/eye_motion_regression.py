from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from vis_vest_hd_computation.imports import *


def plot_r2(lags, score, fname='r2.pdf', fig_path='.'):
    fig, ax = plt.subplots(1,1, figsize=(2.0,1.75), tight_layout=True)
    plt.plot(lags, score)
    plt.xticks(lags)
    plt.xlabel('lag [frames]')
    plt.ylabel('R2')
    fig.savefig(os.path.join(fig_path, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def lag_var(df, var=EYE_ROTATION, lag=0, max_lag=6):
    out = df.groupby(['motor_velocity', 'grating_velocity', 'static_grating_interval']).apply(lambda x: x.iloc[0])[['motor_velocity', 'grating_velocity', 'static_grating_interval']].values
    vm = out[:, 0]
    vg = out[:, 1]
    static_grating_interval = out[:, 2]

    lagged_var = []
    for v1, v2, sgi in zip(vm, vg, static_grating_interval):
        mask = np.logical_and.reduce((df['motor_velocity'] == v1, df['grating_velocity'] == v2, df['static_grating_interval'] == sgi))
        var_data = df[mask][var].to_numpy()
        var_data = np.roll(var_data, -lag)
        if lag > 0:
           var_data[-lag:] = np.nan
           var_data[:max_lag-lag] = np.nan
        lagged_var.append(var_data)

    df['lagged_{}'.format(var)] = np.concatenate(lagged_var)

    return df


def comp_vestibular_signals(df, T=T_TRANSIENT_STIM, fps=FPS_MINISCOPE, tau=TAU_C):

    from vis_vest_hd_computation.model import vestibular_signal, source

    out = df.groupby(['motor_velocity', 'grating_velocity', 'static_grating_interval']).apply(lambda x: x.iloc[0])[['motor_velocity', 'grating_velocity', 'static_grating_interval']].values
    vm = out[:,0]
    vg = out[:,1]
    static_grating_interval = out[:,2]

    k = 60
    t = np.linspace(0,T, int(k*T + 1))
    dt = 1 / k
    skip = int(k // fps)

    c = []
    dc = []
    int_c = []

    for v1, v2, sgi in zip(vm, vg, static_grating_interval):
        ci, dc_i = vestibular_signal(t, v1 * np.pi / 180, T=T, tau=tau)
        c.append(ci[::skip])
        dc.append(dc_i[::skip])
        int_c.append(np.cumsum(ci)[::skip] * dt)

    df['int_c'] = np.concatenate(int_c)
    df['c'] = np.concatenate(c)
    df['dc'] = np.concatenate(dc)

    return df


def plot_rotation_time_series(df, eye_dir_pred, atn_dir_pred, T=3, lag=0, asymmetric=False, fname='rotation_over_time.pdf', figpath='.'):

    import matplotlib.pyplot as plt

    if asymmetric:
        get_mask = lambda df, vm, vg: np.logical_and.reduce((df['motor_velocity'] == vm, df['grating_velocity'] == vg, df['static_grating_interval']>-1))
    else:
        get_mask = lambda df, vm, vg: np.logical_and.reduce((df['motor_velocity'] == vm, df['grating_velocity'] == vg, df['static_grating_interval']==-1))

    motor_vels = df['motor_velocity'].unique()
    grating_vels = df['grating_velocity'].unique()

    fig, axs = plt.subplots(len(grating_vels), len(motor_vels), figsize=(7.24,11), constrained_layout=True)

    for i, vg in enumerate(grating_vels):
        for j, vm in enumerate(motor_vels):
            mask = get_mask(df, vm, vg)
            axs[i,j].plot(df['t'][mask], df[DECODED_ROTATION][mask] * 180 / np.pi, DECODING_COLOR)

            if atn_dir_pred is not None:
                axs[i,j].plot(df['t'][mask], atn_dir_pred[mask] * 180 / np.pi, DECODING_COLOR, linestyle='--')

            axs[i,j].plot(df['t'][mask],  df[NEG_EYE_ROTATION][mask] * 180 / np.pi, EYE_COLOR)

            if eye_dir_pred is not None:
                axs[i,j].plot(df['t'][mask] + lag, -eye_dir_pred[mask] * 180 / np.pi, 'k--')

            axs[i,j].set_title('{}={}, {}={}'.format(HEAD_VAR, vm, SCENE_VAR, vg))
            axs[i,j].set_xlabel('t [s]')
            axs[i,j].set_ylabel('Shift [deg]')
            #axs[i,j].grid(True, which='both')
            axs[i,j].set_xlim(0, T)
            axs[i,j].set_xticks(np.arange(T+1))

    fig.savefig(os.path.join(figpath, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def regression(df, out_path='eye_motion_regression', fig_path='.', tau=TAU_C, fps=FPS_MINISCOPE, asymmetric=False):

    df = df.sort_values(by=['motor_velocity', 'grating_velocity', 'static_grating_interval', 't'], ignore_index=True, inplace=False)
    df = comp_vestibular_signals(df, tau=tau)

    fig_path = os.path.join(fig_path, 'eye_motion_regression') 
    os.makedirs(fig_path, exist_ok=True)

    lags = np.arange(7)
    mse = np.zeros(len(lags))
    r2 = np.zeros(len(lags))
    coef = np.zeros((len(lags), 3))

    for k, lag in enumerate(lags):
        df = lag_var(df, lag=lag)
        X = np.vstack((df[DECODED_ROTATION].to_numpy(), df['int_c'].to_numpy(), df['c'].to_numpy())).T
        y = df['lagged_{}'.format(EYE_ROTATION)].to_numpy().squeeze()

        mask = np.logical_not(np.isnan(y))
        reg = LinearRegression(fit_intercept=False).fit(X[mask], y[mask])
        y_pred = reg.predict(X)

        coef[k,:] = reg.coef_
        r2[k] = reg.score(X[mask], y[mask])
        mse[k] = mean_squared_error(y[mask], y_pred[mask])

        fname = 'rotation_over_time_tau={:1.1f}s_lag={}.pdf'.format(tau, lag)

        plot_rotation_time_series(df, y_pred, None, lag=lag/12, asymmetric=asymmetric, fname=fname, figpath=fig_path)

    save_path = os.path.join(BASE_PATH, 'data', 'trial_data', out_path, 'eye_motion_regression')
    os.makedirs(save_path, exist_ok=True)

    np.savetxt(os.path.join(save_path, 'lags.csv'), lags, delimiter=",")
    np.savetxt(os.path.join(save_path, 'mse.csv'), mse, delimiter=",")
    np.savetxt(os.path.join(save_path, 'r2.csv'), r2, delimiter=",")
    np.savetxt(os.path.join(save_path, 'coefs.csv'), coef)

    k = np.argmax(r2)
    opt_lag = lags[k] / fps
    opt_coef = -coef[k,:]

    np.savetxt(os.path.join(save_path, 'plant_lag.csv'), np.array([opt_lag]), delimiter=",")
    np.savetxt(os.path.join(save_path, 'plant_coef.csv'), opt_coef, delimiter=",")

    plot_r2(lags, r2, fname='r2.pdf', fig_path=fig_path)

    return opt_lag, opt_coef



