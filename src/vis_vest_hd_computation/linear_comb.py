import vis_vest_hd_computation.utils as utils
from vis_vest_hd_computation.imports import *
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def linear_comb(df_restricted, fname='linear_combination.pdf', fig_path='.'):

    df = utils.align_restricted(df_restricted, inplace=False)

    cols = ['stim_type', 'vel']
    df = df.groupby(cols)[DECODED_ROTATION].agg('mean').reset_index()
    df = df[df['vel'] > 0].copy()

    mask_head = df['stim_type'] == 'head'
    mask_scene = df['stim_type'] == 'scene'
    mask_sync = df['stim_type'] == 'sync'

    scene = df[mask_scene]
    head = df[mask_head]
    sync = df[mask_sync]

    ref_shift = T_TRANSIENT_STIM * np.array(head['vel']) / 90

    X = np.vstack((scene[DECODED_ROTATION] / ref_shift, sync[DECODED_ROTATION] / ref_shift)).T
    y = np.array(head[DECODED_ROTATION]) / ref_shift

    model = LinearRegression(fit_intercept=False).fit(X, y)

    # print(f"coefficients: {model.coef_}")

    y_pred = model.predict(X)

    fig = plt.figure(figsize=(1.75, 1.5))
    ax = fig.add_axes([0.3, 0.3, 0.65, 0.6])

    ax.plot(df['vel'][mask_head], 100 * df[DECODED_ROTATION][mask_head] / ref_shift , color='k', marker='.', markersize=7, label='Head-only')
    ax.plot(df['vel'][mask_head], 100 * y_pred , linestyle='dashed', color='k', marker='s', markersize=3, label='Optimal linear combination')
    plt.legend(fontsize=5)

    vels = np.unique(df['vel'])
    ax.set_xlabel('Peak speed [deg/s]')
    ax.set_ylabel('Decoded HD shift [%]')
    ax.set_xlim(20, 205)
    ax.set_xticks(vels)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75,100])

    fig.savefig(os.path.join(fig_path, fname), dpi=300)
    plt.close(fig)
