# %%
import numpy as np
import pandas as pd
import pymap3d as pm
import pymap3d.vincenty as pmv
import matplotlib.pyplot as plt
import glob as gl
import scipy.optimize
from tqdm.auto import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial import distance
import pickle
# Constants
CLIGHT = 299_792_458  # speed of light (m/s)
RE_WGS84 = 6_378_137  # earth semimajor axis (WGS84) (m)
OMGE = 7.2921151467E-5  # earth angular velocity (IS-GPS) (rad/s)


# %% md
# Satellite Selection
# %%
# Satellite selection using carrier frequency error, elevation angle, and C/N0
def satellite_selection(df, column):
    """
    Args:
        df : DataFrame from device_gnss.csv
        column : Column name
    Returns:
        df: DataFrame with eliminated satellite signals
    """
    idx = df[column].notnull()
    idx &= df['CarrierErrorHz'] < 2.0e6  # carrier frequency error (Hz)
    idx &= df['SvElevationDegrees'] > 10.0  # elevation angle (deg)
    idx &= df['Cn0DbHz'] > 15.0  # C/N0 (dB-Hz)
    idx &= df['MultipathIndicator'] == 0  # Multipath flag

    return df[idx]


# %% md
# Pseudorange/Doppler Residuals and Jacobian
# %%
# Compute line-of-sight vector from user to satellite
def los_vector(xusr, xsat):
    """
    Args:
        xusr : user position in ECEF (m)
        xsat : satellite position in ECEF (m)
    Returns:
        u: unit line-of-sight vector in ECEF (m)
        rng: distance between user and satellite (m)
    """
    u = xsat - xusr
    rng = np.linalg.norm(u, axis=1).reshape(-1, 1)
    u /= rng

    return u, rng.reshape(-1)


# Compute Jacobian matrix
def jac_pr_residuals(x, xsat, pr, W):
    """
    Args:
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        pr : pseudorange (m)
        W : weight matrix
    Returns:
        W*J : Jacobian matrix
    """
    u, _ = los_vector(x[:3], xsat)
    J = np.hstack([-u, np.ones([len(pr), 1])])  # J = [-ux -uy -uz 1]

    return W @ J


# Compute pseudorange residuals
def pr_residuals(x, xsat, pr, W):
    """
    Args:
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        pr : pseudorange (m)
        W : weight matrix
    Returns:
        residuals*W : pseudorange residuals
    """
    u, rng = los_vector(x[:3], xsat)

    # Approximate correction of the earth rotation (Sagnac effect) often used in GNSS positioning
    rng += OMGE * (xsat[:, 0] * x[1] - xsat[:, 1] * x[0]) / CLIGHT

    # Add GPS L1 clock offset
    residuals = rng - (pr - x[3])

    return residuals @ W


# Compute Jacobian matrix
def jac_prr_residuals(v, vsat, prr, x, xsat, W):
    """
    Args:
        v : current velocity in ECEF (m/s)
        vsat : satellite velocity in ECEF (m/s)
        prr : pseudorange rate (m/s)
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        W : weight matrix
    Returns:
        W*J : Jacobian matrix
    """
    u, _ = los_vector(x[:3], xsat)
    J = np.hstack([-u, np.ones([len(prr), 1])])

    return W @ J


# Compute pseudorange rate residuals
def prr_residuals(v, vsat, prr, x, xsat, W):
    """
    Args:
        v : current velocity in ECEF (m/s)
        vsat : satellite velocity in ECEF (m/s)
        prr : pseudorange rate (m/s)
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        W : weight matrix
    Returns:
        residuals*W : pseudorange rate residuals
    """
    u, rng = los_vector(x[:3], xsat)
    rate = np.sum((vsat - v[:3]) * u, axis=1) \
           + OMGE / CLIGHT * (vsat[:, 1] * x[0] + xsat[:, 1] * v[0]
                              - vsat[:, 0] * x[1] - xsat[:, 0] * v[1])

    residuals = rate - (prr - v[3])

    return residuals @ W

# Carrier smoothing of pseudarange
def carrier_smoothing(gnss_df):
    """
    Args:
        df : DataFrame from device_gnss.csv
    Returns:
        df: DataFrame with carrier-smoothing pseudorange 'pr_smooth'
    """
    carr_th = 1.5  # carrier phase jump threshold [m] ** 2.0 -> 1.5 **
    pr_th = 20.0  # pseudorange jump threshold [m]

    prsmooth = np.full_like(gnss_df['RawPseudorangeMeters'], np.nan)
    # Loop for each signal
    for (i, (svid_sigtype, df)) in enumerate((gnss_df.groupby(['Svid', 'SignalType']))):
        df = df.replace(
            {'AccumulatedDeltaRangeMeters': {0: np.nan}})  # 0 to NaN

        # Compare time difference between pseudorange/carrier with Doppler
        drng1 = df['AccumulatedDeltaRangeMeters'].diff() - df['PseudorangeRateMetersPerSecond']
        drng2 = df['RawPseudorangeMeters'].diff() - df['PseudorangeRateMetersPerSecond']

        # Check cycle-slip
        slip1 = (df['AccumulatedDeltaRangeState'].to_numpy() & 2 ** 1) != 0  # reset flag
        slip2 = (df['AccumulatedDeltaRangeState'].to_numpy() & 2 ** 2) != 0  # cycle-slip flag
        slip3 = np.fabs(drng1.to_numpy()) > carr_th  # Carrier phase jump
        slip4 = np.fabs(drng2.to_numpy()) > pr_th  # Pseudorange jump

        idx_slip = slip1 | slip2 | slip3 | slip4
        idx_slip[0] = True

        # groups with continuous carrier phase tracking
        df['group_slip'] = np.cumsum(idx_slip)

        # Psudorange - carrier phase
        df['dpc'] = df['RawPseudorangeMeters'] - df['AccumulatedDeltaRangeMeters']

        # Absolute distance bias of carrier phase
        meandpc = df.groupby('group_slip')['dpc'].mean()
        df = df.merge(meandpc, on='group_slip', suffixes=('', '_Mean'))

        # Index of original gnss_df
        idx = (gnss_df['Svid'] == svid_sigtype[0]) & (
                gnss_df['SignalType'] == svid_sigtype[1])

        # Carrier phase + bias
        prsmooth[idx] = df['AccumulatedDeltaRangeMeters'] + df['dpc_Mean']

    # If carrier smoothing is not possible, use original pseudorange
    idx_nan = np.isnan(prsmooth)
    prsmooth[idx_nan] = gnss_df['RawPseudorangeMeters'][idx_nan]
    gnss_df['pr_smooth'] = prsmooth

    return gnss_df


# %% md
# Score Computation
# %%
# Compute distance by Vincenty's formulae
def vincenty_distance(llh1, llh2):
    """
    Args:
        llh1 : [latitude,longitude] (deg)
        llh2 : [latitude,longitude] (deg)
    Returns:
        d : distance between llh1 and llh2 (m)
    """
    d, az = np.array(pmv.vdist(llh1[:, 0], llh1[:, 1], llh2[:, 0], llh2[:, 1]))

    return d


# Compute score
def calc_score(llh, llh_gt):
    """
    Args:
        llh : [latitude,longitude] (deg)
        llh_gt : [latitude,longitude] (deg)
    Returns:
        score : (m)
    """
    d = vincenty_distance(llh, llh_gt)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])

    return score

def calc_score_rnan(llh, llh_gt):
    """
    Args:
        llh : [latitude,longitude] (deg)
        llh_gt : [latitude,longitude] (deg)
    Returns:
        score : (m)
    """
    pos = ~np.isnan(llh[:, 0])
    d = vincenty_distance(llh[pos, :], llh_gt[pos, :])
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])

    return score

# GNSS single point positioning using pseudorange
def point_positioning(gnss_df):
    # Add nominal frequency to each signal
    # Note: GLONASS is an FDMA signal, so each satellite has a different frequency
    CarrierFrequencyHzRef = gnss_df.groupby(['Svid', 'SignalType'])[
        'CarrierFrequencyHz'].median()
    gnss_df = gnss_df.merge(CarrierFrequencyHzRef, how='left', on=[
        'Svid', 'SignalType'], suffixes=('', 'Ref'))
    gnss_df['CarrierErrorHz'] = np.abs(
        (gnss_df['CarrierFrequencyHz'] - gnss_df['CarrierFrequencyHzRef']))

    # Carrier smoothing
    gnss_df = carrier_smoothing(gnss_df)

    # GNSS single point positioning
    utcTimeMillis = gnss_df['utcTimeMillis'].unique()
    nepoch = len(utcTimeMillis)
    x0 = np.zeros(4)  # [x,y,z,tGPSL1]
    v0 = np.zeros(4)  # [vx,vy,vz,dtGPSL1]
    # x_wls = np.full([nepoch, 3], np.nan)  # For saving position
    # v_wls = np.full([nepoch, 3], np.nan)  # For saving velocity
    x_wls = np.full([nepoch, 3], 0.0)  # For saving position
    # v_wls = np.full([nepoch, 3], 0.0)  # For saving velocity
    v_wls = np.full([nepoch, 3], np.nan)  # For saving velocity from 0 to nan init
    cov_x = np.full([nepoch, 3, 3], np.nan)  # For saving position covariance
    cov_v = np.full([nepoch, 3, 3], np.nan)  # For saving velocity covariance

    # Loop for epochs
    for i, (t_utc, df) in enumerate(tqdm(gnss_df.groupby('utcTimeMillis'), total=nepoch)):
        # Valid satellite selection
        df_pr = satellite_selection(df, 'pr_smooth')
        df_prr = satellite_selection(df, 'PseudorangeRateMetersPerSecond')

        # Corrected pseudorange/pseudorange rate
        pr = (df_pr['pr_smooth'] + df_pr['SvClockBiasMeters'] - df_pr['IsrbMeters'] -
              df_pr['IonosphericDelayMeters'] - df_pr['TroposphericDelayMeters']).to_numpy()
        prr = (df_prr['PseudorangeRateMetersPerSecond'] +
               df_prr['SvClockDriftMetersPerSecond']).to_numpy()

        # Satellite position/velocity
        xsat_pr = df_pr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                         'SvPositionZEcefMeters']].to_numpy()
        xsat_prr = df_prr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                           'SvPositionZEcefMeters']].to_numpy()
        vsat = df_prr[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                       'SvVelocityZEcefMetersPerSecond']].to_numpy()
        # Weight matrix for peseudorange/pseudorange rate
        Wx = np.diag(1 / df_pr['RawPseudorangeUncertaintyMeters'].to_numpy())
        Wv = np.diag(1 / df_prr['PseudorangeRateUncertaintyMetersPerSecond'].to_numpy())

        # Robust WLS requires accurate initial values for convergence,
        # so perform normal WLS for the first time
        if len(df_pr) >= 4:
            # Normal WLS
            if np.all(x0 == 0):
                opt = scipy.optimize.least_squares(
                    pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx))
                x0 = opt.x
                # Robust WLS for position estimation
            opt = scipy.optimize.least_squares(
                pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx), loss='soft_l1')
            if opt.status < 1 or opt.status == 2:
                print(f'i = {i} position lsq status = {opt.status}')
            else:
                # Covariance estimation
                cov = np.linalg.inv(opt.jac.T @ Wx @ opt.jac) # 什么原理？出处在哪
                cov_x[i, :, :] = cov[:3, :3]
                x_wls[i, :] = opt.x[:3]
                x0 = opt.x

        # Velocity estimation
        if len(df_prr) >= 4:
            if np.all(v0 == 0):  # Normal WLS
                opt = scipy.optimize.least_squares(
                    prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv))
                v0 = opt.x
            # Robust WLS for velocity estimation
            opt = scipy.optimize.least_squares(
                prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv), loss='soft_l1')
            if opt.status < 1:
                print(f'i = {i} velocity lsq status = {opt.status}')
            else:
                # Covariance estimation
                cov = np.linalg.inv(opt.jac.T @ Wv @ opt.jac)
                cov_v[i, :, :] = cov[:3, :3]
                v_wls[i, :] = opt.x[:3]
                v0 = opt.x

    return utcTimeMillis, x_wls, v_wls, cov_x, cov_v


# %% md
# Outlier Detection and Interpolation
# %%
# Simple outlier detection and interpolation
def exclude_interpolate_outlier(x_wls, v_wls, cov_x, cov_v):
    # Up velocity / height threshold
    v_up_th = 2.6  # m/s  2.0 -> 2.6
    height_th = 200.0  # m
    v_out_sigma = 3.0  # m/s
    x_out_sigma = 30.0  # m

    # Coordinate conversion
    x_llh = np.array(pm.ecef2geodetic(x_wls[:, 0], x_wls[:, 1], x_wls[:, 2])).T
    x_llh_mean = np.nanmean(x_llh, axis=0)
    v_enu = np.array(pm.ecef2enuv(
        v_wls[:, 0], v_wls[:, 1], v_wls[:, 2], x_llh_mean[0], x_llh_mean[1])).T

    # Up velocity jump detection
    # Cars don't jump suddenly!
    idx_v_out = np.abs(v_enu[:, 2]) > v_up_th
    idx_v_out |= np.isnan(v_enu[:, 2])
    v_wls[idx_v_out, :] = np.nan
    cov_v[idx_v_out] = v_out_sigma ** 2 * np.eye(3)
    print(f'Number of velocity outliers {np.count_nonzero(idx_v_out)}')

    # Height check
    hmedian = np.nanmedian(x_llh[:, 2])
    idx_x_out = np.abs(x_llh[:, 2] - hmedian) > height_th
    idx_x_out |= np.isnan(x_llh[:, 2])
    x_wls[idx_x_out, :] = np.nan
    cov_x[idx_x_out] = x_out_sigma ** 2 * np.eye(3)
    print(f'Number of position outliers {np.count_nonzero(idx_x_out)}')

    # Interpolate NaNs at beginning and end of array
    x_df = pd.DataFrame({'x': x_wls[:, 0], 'y': x_wls[:, 1], 'z': x_wls[:, 2]})
    x_df = x_df.interpolate(limit_area='outside', limit_direction='both')

    # Interpolate all NaN data
    v_df = pd.DataFrame({'x': v_wls[:, 0], 'y': v_wls[:, 1], 'z': v_wls[:, 2]})
    v_df = v_df.interpolate(limit_area='outside', limit_direction='both')
    v_df = v_df.interpolate('spline', order=3)

    return x_df.to_numpy(), v_df.to_numpy(), cov_x, cov_v


def Kalman_filter(zs, us, cov_zs, cov_us, phone):
    # Parameters
    sigma_mahalanobis = 30.0  # Mahalanobis distance for rejecting innovation

    n, dim_x = zs.shape
    F = np.eye(3)  # Transition matrix
    H = np.eye(3)  # Measurement function

    # Initial state and covariance
    x = zs[0, :3].T  # State
    P = 5.0 ** 2 * np.eye(3)  # State covariance
    I = np.eye(dim_x)

    x_kf = np.zeros([n, dim_x])
    P_kf = np.zeros([n, dim_x, dim_x])

    # Kalman filtering
    for i, (u, z) in enumerate(zip(us, zs)):
        # First step
        if i == 0:
            x_kf[i] = x.T
            P_kf[i] = P
            continue

        # Prediction step
        Q = cov_us[i]  # Estimated WLS velocity covariance
        x = F @ x + u.T
        P = (F @ P) @ F.T + Q

        # Check outliers for observation
        d = distance.mahalanobis(z, H @ x, np.linalg.inv(P))

        # Update step
        if d < sigma_mahalanobis:
            R = cov_zs[i]  # Estimated WLS position covariance
            y = z.T - H @ x
            S = (H @ P) @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)
            x = x + K @ y
            P = (I - (K @ H)) @ P
        else:
            # If observation update is not available, increase covariance
            P += 10 ** 2 * Q

        x_kf[i] = x.T
        P_kf[i] = P

    return x_kf, P_kf


# Forward + backward Kalman filter and smoothing
def Kalman_smoothing(x_wls, v_wls, cov_x, cov_v, phone):
    n, dim_x = x_wls.shape

    # For some unknown reason, the speed estimation is wrong only for XiaomiMi8
    # so the variance is increased
    if phone == 'XiaomiMi8':
        v_wls = np.vstack([(v_wls[:-1, :] + v_wls[1:, :]) / 2, np.zeros([1, 3])])
        cov_v = 1000.0 ** 2 * cov_v

    # Forward
    v = np.vstack([np.zeros([1, 3]), (v_wls[:-1, :] + v_wls[1:, :]) / 2])
    x_f, P_f = Kalman_filter(x_wls, v, cov_x, cov_v, phone)

    # Backward
    v = -np.flipud(v_wls)
    v = np.vstack([np.zeros([1, 3]), (v[:-1, :] + v[1:, :]) / 2])
    cov_xf = np.flip(cov_x, axis=0)
    cov_vf = np.flip(cov_v, axis=0)
    x_b, P_b = Kalman_filter(np.flipud(x_wls), v, cov_xf, cov_vf, phone)

    # Smoothing
    x_fb = np.zeros_like(x_f)
    P_fb = np.zeros_like(P_f)
    for (f, b) in zip(range(n), range(n - 1, -1, -1)):
        P_fi = np.linalg.inv(P_f[f])
        P_bi = np.linalg.inv(P_b[b])

        P_fb[f] = np.linalg.inv(P_fi + P_bi)
        x_fb[f] = P_fb[f] @ (P_fi @ x_f[f] + P_bi @ x_b[b])

    return x_fb, x_f, np.flipud(x_b)

def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * RADIUS * np.arcsin(a ** 0.5)
    return dist

def correct_outliers(df, th=2):
    df['dist_pre'] = 0
    df['dist_pro'] = 0

    df['latDeg_pre'] = df['LatitudeDegrees'].shift(periods=1, fill_value=0)
    df['lngDeg_pre'] = df['LongitudeDegrees'].shift(periods=1, fill_value=0)
    df['latDeg_pro'] = df['LatitudeDegrees'].shift(periods=-1, fill_value=0)
    df['lngDeg_pro'] = df['LongitudeDegrees'].shift(periods=-1, fill_value=0)
    df['dist_pre'] = calc_haversine(df.latDeg_pre, df.lngDeg_pre, df.LatitudeDegrees, df.LongitudeDegrees)
    df['dist_pro'] = calc_haversine(df.LatitudeDegrees, df.LongitudeDegrees, df.latDeg_pro, df.lngDeg_pro)

    df.loc[df.index.min(), 'dist_pre'] = 0
    df.loc[df.index.max(), 'dist_pro'] = 0

    pro_95 = df['dist_pro'].mean() + (df['dist_pro'].std() * th)
    pre_95 = df['dist_pre'].mean() + (df['dist_pre'].std() * th)

    ind = df[(df['dist_pro'] > pro_95) & (df['dist_pre'] > pre_95)][['dist_pre', 'dist_pro']].index

    for i in ind:
        df.loc[i, 'LatitudeDegrees'] = (df.loc[i - 1, 'LatitudeDegrees'] + df.loc[i + 1, 'LatitudeDegrees']) / 2
        df.loc[i, 'LongitudeDegrees'] = (df.loc[i - 1, 'LongitudeDegrees'] + df.loc[i + 1, 'LongitudeDegrees']) / 2

    return df

def remove_nans(x_wls):
    nan_pos = np.argwhere(np.isnan(np.reshape(x_wls[:, 0], [len(x_wls[:, 0]), 1])))[:, 0]
    numnan = len(nan_pos)  # np.sum(np.isnan(x_wls))
    len_x = len(x_wls)
    x_wls_rnan = x_wls.copy()
    if numnan > 0 and numnan < 0.1 * len_x:
        cnt = 1
        for pos in nan_pos:
            if cnt < numnan:
                if pos > 2 and nan_pos[cnt] - pos > 2:
                    x_wls_rnan[pos, :] = (x_wls_rnan[pos - 2, :] + x_wls_rnan[pos - 1, :] + x_wls_rnan[pos + 1,
                                                                                            :] + x_wls_rnan[pos + 2,
                                                                                                 :]) / 4
                elif pos > 1 and nan_pos[cnt] - pos > 1:
                    x_wls_rnan[pos, :] = (x_wls_rnan[pos - 1, :] + x_wls_rnan[pos + 1, :]) / 2
            else:
                if pos > 2 and pos < len_x - 2:
                    x_wls_rnan[pos, :] = (x_wls_rnan[pos - 2, :] + x_wls_rnan[pos - 1, :] + x_wls_rnan[pos + 1,
                                                                                            :] + x_wls_rnan[pos + 2,
                                                                                                 :]) / 4
                elif pos > 1 and pos < len_x - 1:
                    x_wls_rnan[pos, :] = (x_wls_rnan[pos - 1, :] + x_wls_rnan[pos + 1, :]) / 2
            cnt += 1
    print(f'orig nans: {numnan}, removed nans to {np.sum(np.isnan(x_wls_rnan)) / np.shape(x_wls)[1]}')
    return x_wls_rnan

# Coordinate conversions (From https://github.com/commaai/laika)
a = 6378137
b = 6356752.3142
esq = 6.69437999014 * 0.001
e1sq = 6.73949674228 * 0.001

def geodetic2ecef(geodetic, radians=False):
    geodetic = np.array(geodetic)
    input_shape = geodetic.shape
    geodetic = np.atleast_2d(geodetic)

    ratio = 1.0 if radians else (np.pi / 180.0)
    lat = ratio * geodetic[:, 0]
    lon = ratio * geodetic[:, 1]
    alt = geodetic[:, 2]

    xi = np.sqrt(1 - esq * np.sin(lat) ** 2)
    x = (a / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (a / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (a / xi * (1 - esq) + alt) * np.sin(lat)
    ecef = np.array([x, y, z]).T
    return ecef.reshape(input_shape)

# GNSS single point positioning using pseudorange
def point_positioning_reinit(gnss_df):
    # Add nominal frequency to each signal
    # Note: GLONASS is an FDMA signal, so each satellite has a different frequency
    CarrierFrequencyHzRef = gnss_df.groupby(['Svid', 'SignalType'])[
        'CarrierFrequencyHz'].median()
    gnss_df = gnss_df.merge(CarrierFrequencyHzRef, how='left', on=[
        'Svid', 'SignalType'], suffixes=('', 'Ref'))
    gnss_df['CarrierErrorHz'] = np.abs(
        (gnss_df['CarrierFrequencyHz'] - gnss_df['CarrierFrequencyHzRef']))

    # Carrier smoothing
    gnss_df = carrier_smoothing(gnss_df)

    # GNSS single point positioning
    utcTimeMillis = gnss_df['utcTimeMillis'].unique()
    nepoch = len(utcTimeMillis)
    x0 = np.zeros(4)  # [x,y,z,tGPSL1]
    v0 = np.zeros(4)  # [vx,vy,vz,dtGPSL1]
    # x_wls = np.full([nepoch, 3], 0.0)  # For saving position
    # v_wls = np.full([nepoch, 3], 0.0)  # For saving velocity
    x_wls = np.full([nepoch, 3], np.nan)  # For saving position from 0 to nan init
    v_wls = np.full([nepoch, 3], np.nan)  # For saving velocity from 0 to nan init
    x_wls_igst = np.full([nepoch, 3], np.nan)  # For saving position from 0 to nan init
    v_wls_igst = np.full([nepoch, 3], np.nan)  # For saving velocity from 0 to nan init
    cov_x = np.full([nepoch, 3, 3], np.nan)  # For saving position covariance
    cov_v = np.full([nepoch, 3, 3], np.nan)  # For saving velocity covariance

    # Loop for epochs
    for i, (t_utc, df) in enumerate(tqdm(gnss_df.groupby('utcTimeMillis'), total=nepoch)):
        # Valid satellite selection
        df_pr = satellite_selection(df, 'pr_smooth')
        df_prr = satellite_selection(df, 'PseudorangeRateMetersPerSecond')

        # Corrected pseudorange/pseudorange rate
        pr = (df_pr['pr_smooth'] + df_pr['SvClockBiasMeters'] - df_pr['IsrbMeters'] -
              df_pr['IonosphericDelayMeters'] - df_pr['TroposphericDelayMeters']).to_numpy()
        prr = (df_prr['PseudorangeRateMetersPerSecond'] +
               df_prr['SvClockDriftMetersPerSecond']).to_numpy()

        # Satellite position/velocity
        xsat_pr = df_pr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                         'SvPositionZEcefMeters']].to_numpy()
        xsat_prr = df_prr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                           'SvPositionZEcefMeters']].to_numpy()
        vsat = df_prr[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                       'SvVelocityZEcefMetersPerSecond']].to_numpy()

        # Weight matrix for peseudorange/pseudorange rate
        Wx = np.diag(1 / df_pr['RawPseudorangeUncertaintyMeters'].to_numpy())
        Wv = np.diag(1 / df_prr['PseudorangeRateUncertaintyMetersPerSecond'].to_numpy())

        # Robust WLS requires accurate initial values for convergence,
        # so perform normal WLS for the first time
        if len(df_pr) >= 4:
            # Normal WLS
            if np.all(x0 == 0):
                opt = scipy.optimize.least_squares(
                    pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx))
                x0 = opt.x
            # Robust WLS for position estimation
            opt = scipy.optimize.least_squares(
                pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx), loss='soft_l1')
            if opt.status < 1 or opt.status == 2:
                print(f'i = {i} position lsq status = {opt.status}')
                x_wls_igst[i, :] = opt.x[:3]
            else:
                # Covariance estimation
                cov = np.linalg.inv(opt.jac.T @ Wx @ opt.jac)
                cov_x[i, :, :] = cov[:3, :3]
                x_wls[i, :] = opt.x[:3]
                x_wls_igst[i, :] = opt.x[:3]
                x0 = opt.x

        # Velocity estimation
        if len(df_prr) >= 4:
            if np.all(v0 == 0):  # Normal WLS
                opt = scipy.optimize.least_squares(
                    prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv))
                v0 = opt.x
            # Robust WLS for velocity estimation
            opt = scipy.optimize.least_squares(
                prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv), loss='soft_l1')
            if opt.status < 1:
                print(f'i = {i} velocity lsq status = {opt.status}')
                v_wls_igst[i, :] = opt.x[:3]
            else:
                # Covariance estimation
                cov = np.linalg.inv(opt.jac.T @ Wv @ opt.jac)
                cov_v[i, :, :] = cov[:3, :3]
                v_wls[i, :] = opt.x[:3]
                v_wls_igst[i, :] = opt.x[:3]
                v0 = opt.x

    return utcTimeMillis, x_wls, v_wls, cov_x, cov_v, x_wls_igst, v_wls_igst


import os
from pathlib import Path
import pandas as pd

# path = '/home/suwensheng/goo_com/competitions/google-smartphone-decimeter-challenge'
# 10.23.18.26
path = '/mnt/sdb/home/tangjh/smartphone-decimeter-2022' # 修改使用者的项目绝对路径
savepath = Path('/mnt/sdb/home/tangjh/smartphone-decimeter-2022')

test_dfs = []
truth_dfs = []
record_dfs = []

# if os.path.exists(savepath/'baseline_locations_train_2022_ecef_igst.csv'):
#     sample_df = pd.read_csv(savepath/'baseline_locations_train_2022_ecef_igst.csv')
#     test_dfs.append(sample_df)
#
# if os.path.exists(savepath/'records_scores_train_2022_ecef_igst.csv'):
#     sample_df = pd.read_csv(savepath/'records_scores_train_2022_ecef_igst.csv')
#     record_dfs.append(sample_df)
# sample_df = pd.read_csv(f'{path}/baseline_locations_train.csv')
# KF nan skip list
tripIDskiplist = ['2020-08-03-US-MTV-2/GooglePixel5']
nullaltitude2022list = ['2021-01-05-US-MTV-1', '2021-01-05-US-MTV-2']
zeroobs2022trajs = ['2021-03-16-US-MTV-3/XiaomiMi8', '2021-04-02-US-SJC-1/GooglePixel4',
                    '2021-04-02-US-SJC-1/XiaomiMi8',
                    '2021-04-26-US-SVL-2/SamsungGalaxyS20Ultra', '2021-04-26-US-SVL-2/XiaomiMi8',
                    '2021-07-14-US-MTV-1/GooglePixel5',
                    '2021-07-14-US-MTV-1/SamsungGalaxyS20Ultra', '2021-07-14-US-MTV-1/XiaomiMi8',
                    '2021-07-27-US-MTV-1/GooglePixel5',
                    '2021-12-07-US-LAX-1/GooglePixel5', '2021-12-07-US-LAX-2/GooglePixel5',
                    '2021-12-09-US-LAX-2/GooglePixel5',
                    '2021-07-19-US-MTV-1/XiaomiMi8', '2021-08-24-US-SVL-1/SamsungGalaxyS20Ultra']
# Loop for each trip
covx_dic,covv_dic = {},{} # to record covariance of position and velocity
trip_wls_igst_null = ['2021-04-29-US-MTV-1/SamsungGalaxyS20Ultra'] # '2021-04-29-US-MTV-1/SamsungGalaxyS20Ultra','2021-07-01-US-MTV-1/XiaomiMi8','2021-12-07-US-LAX-1/XiaomiMi8','2021-12-07-US-LAX-2/XiaomiMi8','2021-12-08-US-LAX-1/GooglePixel6Pro','2021-12-09-US-LAX-2/XiaomiMi8'


for i, dirname in enumerate(tqdm(sorted(gl.glob(f'{path}/train/*/*/')))):
    drive, phone = dirname.split('/')[-3:-1]
    tripID = f'{drive}/{phone}'
    if tripID not in trip_wls_igst_null:
        continue
    if '2021' in drive:
        # ignoring trajs without altitude
        if drive not in nullaltitude2022list and tripID not in zeroobs2022trajs:
            print(tripID)
            if (not os.path.exists(f'{dirname}/baseline_ecef_igst1.csv')):
                # Read data
                gnss_df = pd.read_csv(f'{dirname}/device_gnss.csv')
                truth_df = pd.read_csv(f'{dirname}/ground_truth.csv')

                # Point positioning
                # utc, x_wls_raw, v_wls_raw, cov_x, cov_v = point_positioning(gnss_df)
                utc, x_wls_raw, v_wls_raw, cov_x, cov_v, x_wls_igst, v_wls_igst = point_positioning_reinit(gnss_df)

                # Exclude velocity outliers
                x_wls, v_wls, cov_x, cov_v = exclude_interpolate_outlier(x_wls_raw, v_wls_raw, cov_x, cov_v)

                # Kalman smoothing
                x_kf, x_f, _ = Kalman_smoothing(x_wls, v_wls, cov_x, cov_v, phone)
                assert np.all(~np.isnan(x_kf))

                # replace limited nans
                x_wls_igst = remove_nans(x_wls)
                v_wls_igst = remove_nans(v_wls)

                x_kf_igst, x_f_igst, _ = Kalman_smoothing(x_wls_igst, v_wls_igst, cov_x, cov_v, phone)
                assert np.all(~np.isnan(x_kf_igst))

                # 填补null
                for i in range(x_wls_igst.shape[0]):
                    row = x_wls_igst[i]
                    for j in range(0, len(row)):
                        # 如果当前元素是 null，则用前一个非 null 元素来填补
                        if np.isnan(row[j]):
                            x_wls_igst[i,j] = x_wls_igst[i-1,j]

                # Baseline
                x_bl = gnss_df.groupby('TimeNanos')[
                    ['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].mean().to_numpy()
                llh_bl = np.array(pm.ecef2geodetic(x_bl[:, 0], x_bl[:, 1], x_bl[:, 2])).T

                # Convert to latitude and longitude
                llh_wls = np.array(pm.ecef2geodetic(x_wls[:, 0], x_wls[:, 1], x_wls[:, 2])).T
                llh_wls_igst = np.array(pm.ecef2geodetic(x_wls_igst[:, 0], x_wls_igst[:, 1], x_wls_igst[:, 2])).T
                llh_kf = np.array(pm.ecef2geodetic(x_kf[:, 0], x_kf[:, 1], x_kf[:, 2])).T
                llh_kf_igst = np.array(pm.ecef2geodetic(x_kf_igst[:, 0], x_kf_igst[:, 1], x_kf_igst[:, 2])).T

                # Ground truth
                llh_gt = truth_df[['LatitudeDegrees', 'LongitudeDegrees']].to_numpy()
                lla_gt = truth_df[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy()
                x_gt = geodetic2ecef(lla_gt)

                # distance in ecef from gt
                decef_bl = np.sqrt(np.sum((x_gt - x_bl) ** 2, axis=1))
                decef_wls = np.sqrt(np.sum((x_gt - x_wls) ** 2, axis=1))
                decef_wls_igst = np.sqrt(np.sum((x_gt - x_wls) ** 2, axis=1))
                decef_kf = np.sqrt(np.sum((x_gt - x_kf) ** 2, axis=1))
                decef_kf_igst = np.sqrt(np.sum((x_gt - x_kf_igst) ** 2, axis=1))

                # Distance from ground truth
                vd_bl = vincenty_distance(llh_bl, llh_gt)
                vd_wls = vincenty_distance(llh_wls, llh_gt)
                vd_wls_igst = vincenty_distance(llh_wls_igst, llh_gt)
                vd_kf = vincenty_distance(llh_kf, llh_gt)
                vd_kf_igst = vincenty_distance(llh_kf_igst, llh_gt)

                # Score
                score_bl = calc_score(llh_bl, llh_gt)
                # score_wls = calc_score(llh_wls, llh_gt)
                # score_wls_igst = calc_score(llh_wls_igst, llh_gt)
                score_wls = calc_score_rnan(llh_wls, llh_gt)
                score_wls_igst = calc_score_rnan(llh_wls_igst, llh_gt)
                score_kf = calc_score(llh_kf[:-1, :], llh_gt[:-1, :])
                score_kf_igst = calc_score(llh_kf_igst[:-1, :], llh_gt[:-1, :])


                # Interpolation for submission
                # UnixTimeMillis = truth_df[truth_df['tripId'] == tripID]['UnixTimeMillis'].to_numpy()
                UnixTimeMillis = truth_df['UnixTimeMillis'].to_numpy()
                lat = InterpolatedUnivariateSpline(utc, llh_kf[:, 0], ext=3)(UnixTimeMillis)
                lng = InterpolatedUnivariateSpline(utc, llh_kf[:, 1], ext=3)(UnixTimeMillis)
                lat_igst = InterpolatedUnivariateSpline(utc, llh_kf_igst[:, 0], ext=3)(UnixTimeMillis)
                lng_igst = InterpolatedUnivariateSpline(utc, llh_kf_igst[:, 1], ext=3)(UnixTimeMillis)
                lat_bl = InterpolatedUnivariateSpline(utc, llh_bl[:, 0], ext=3)(UnixTimeMillis)
                lng_bl = InterpolatedUnivariateSpline(utc, llh_bl[:, 1], ext=3)(UnixTimeMillis)
                lat_wls = InterpolatedUnivariateSpline(utc, llh_wls[:, 0], ext=3)(UnixTimeMillis)
                lng_wls = InterpolatedUnivariateSpline(utc, llh_wls[:, 1], ext=3)(UnixTimeMillis)
                lat_wls_igst = InterpolatedUnivariateSpline(utc, llh_wls_igst[:, 0], ext=3)(UnixTimeMillis)
                lng_wls_igst = InterpolatedUnivariateSpline(utc, llh_wls_igst[:, 1], ext=3)(UnixTimeMillis)
                X_bl = InterpolatedUnivariateSpline(utc, x_bl[:, 0], ext=3)(UnixTimeMillis)
                Y_bl = InterpolatedUnivariateSpline(utc, x_bl[:, 1], ext=3)(UnixTimeMillis)
                Z_bl = InterpolatedUnivariateSpline(utc, x_bl[:, 2], ext=3)(UnixTimeMillis)
                X_wls = InterpolatedUnivariateSpline(utc, x_wls[:, 0], ext=3)(UnixTimeMillis)
                Y_wls = InterpolatedUnivariateSpline(utc, x_wls[:, 1], ext=3)(UnixTimeMillis)
                Z_wls = InterpolatedUnivariateSpline(utc, x_wls[:, 2], ext=3)(UnixTimeMillis)
                X_wls_igst = InterpolatedUnivariateSpline(utc, x_wls_igst[:, 0], ext=3)(UnixTimeMillis)
                Y_wls_igst = InterpolatedUnivariateSpline(utc, x_wls_igst[:, 1], ext=3)(UnixTimeMillis)
                Z_wls_igst = InterpolatedUnivariateSpline(utc, x_wls_igst[:, 2], ext=3)(UnixTimeMillis)
                X_kf = InterpolatedUnivariateSpline(utc, x_kf[:, 0], ext=3)(UnixTimeMillis)
                Y_kf = InterpolatedUnivariateSpline(utc, x_kf[:, 1], ext=3)(UnixTimeMillis)
                Z_kf = InterpolatedUnivariateSpline(utc, x_kf[:, 2], ext=3)(UnixTimeMillis)
                X_kf_igst = InterpolatedUnivariateSpline(utc, x_kf_igst[:, 0], ext=3)(UnixTimeMillis)
                Y_kf_igst = InterpolatedUnivariateSpline(utc, x_kf_igst[:, 1], ext=3)(UnixTimeMillis)
                Z_kf_igst = InterpolatedUnivariateSpline(utc, x_kf_igst[:, 2], ext=3)(UnixTimeMillis)
                # record df for v of wls (modified in 240304)
                vX_wls_igst = InterpolatedUnivariateSpline(utc, v_wls_igst[:, 0], ext=3)(UnixTimeMillis)
                vY_wls_igst = InterpolatedUnivariateSpline(utc, v_wls_igst[:, 1], ext=3)(UnixTimeMillis)
                vZ_wls_igst = InterpolatedUnivariateSpline(utc, v_wls_igst[:, 2], ext=3)(UnixTimeMillis)
                # record df for x of kf_realtime (modified in 240305)
                X_f_igst = InterpolatedUnivariateSpline(utc, x_f_igst[:, 0], ext=3)(UnixTimeMillis)
                Y_f_igst = InterpolatedUnivariateSpline(utc, x_f_igst[:, 1], ext=3)(UnixTimeMillis)
                Z_f_igst = InterpolatedUnivariateSpline(utc, x_f_igst[:, 2], ext=3)(UnixTimeMillis)
                # trip_vel_df = pd.DataFrame({
                #     'tripId': tripID,'UnixTimeMillis': UnixTimeMillis,
                #     'VXEcefMeters_wls_igst': vX_wls_igst, 'VYEcefMeters_wls_igst': vY_wls_igst,
                #     'VZEcefMeters_wls_igst': vZ_wls_igst})
                # trip_vel_df.to_csv(f'{dirname}/Velocity_ecef_igst.csv', index=False)
                trip_kf_realtime_df = pd.DataFrame({
                    'tripId': tripID, 'UnixTimeMillis': UnixTimeMillis,
                    'XEcefMeters_KF_realtime': X_f_igst, 'YEcefMeters_KF_realtime': Y_f_igst,
                    'ZEcefMeters_KF_realtime': Z_f_igst})
                trip_kf_realtime_df.to_csv(f'{dirname}/KF_ecef_igst_realtime.csv', index=False)
                # with open(dirname + '/processed_covariance_velocity.pkl','wb') as value_file:
                #     pickle.dump(cov_v, value_file, True)
                # value_file.close()
                # with open(dirname + '/processed_covariance_position.pkl','wb') as value_file:
                #     pickle.dump(cov_x, value_file, True)
                # value_file.close()

                trip_df = pd.DataFrame({
                    'tripId': tripID,
                    'UnixTimeMillis': UnixTimeMillis,
                    'LatitudeDegrees_bl': lat_bl, 'LongitudeDegrees_bl': lng_bl,
                    'LatitudeDegrees_wls': lat_wls, 'LongitudeDegrees_wls': lng_wls,
                    'LatitudeDegrees_wls_igst': lat_wls_igst, 'LongitudeDegrees_wls_igst': lng_wls_igst,
                    'LatitudeDegrees_kf': lat, 'LongitudeDegrees_kf': lng,
                    'LatitudeDegrees_kf_igst': lat_igst, 'LongitudeDegrees_kf_igst': lng_igst,
                    'XEcefMeters_bl': X_bl, 'YEcefMeters_bl': Y_bl, 'ZEcefMeters_bl': Z_bl,
                    'XEcefMeters_wls': X_wls, 'YEcefMeters_wls': Y_wls, 'ZEcefMeters_wls': Z_wls,
                    'XEcefMeters_wls_igst': X_wls_igst, 'YEcefMeters_wls_igst': Y_wls_igst,
                    'ZEcefMeters_wls_igst': Z_wls_igst,
                    'XEcefMeters_kf': X_kf, 'YEcefMeters_kf': Y_kf, 'ZEcefMeters_kf': Z_kf,
                    'XEcefMeters_kf_igst': X_kf_igst, 'YEcefMeters_kf_igst': Y_kf_igst,
                    'ZEcefMeters_kf_igst': Z_kf_igst,
                })

                test_dfs.append(trip_df)
                # Write submission.csv
                trip_df.to_csv(f'{dirname}/baseline_ecef_igst.csv', index=False)
                test_dftmp = pd.concat(test_dfs)
                # test_dftmp.to_csv(savepath / 'baseline_locations_train_2022_ecef_igst.csv', index=False)
                Distlatlon_wls_avg = vd_wls[~np.isnan(vd_wls)].mean()
                Distlatlon_wls_std = vd_wls[~np.isnan(vd_wls)].std()
                Distecef_wls_avg = decef_wls[~np.isnan(decef_wls)].mean()
                Distecef_wls_std = decef_wls[~np.isnan(decef_wls)].std()

                record_df = pd.DataFrame({
                    'tripId': tripID,
                    'Distlatlon_bl_avg': vd_bl.mean(), 'Distlatlon_bl_std': vd_bl.std(),
                    'Distlatlon_wls_avg': Distlatlon_wls_avg, 'Distlatlon_wls_std': Distlatlon_wls_std,
                    'Distlatlon_wls_igst_avg': vd_wls_igst[~np.isnan(vd_wls_igst)].mean(),
                    'Distlatlon_wls_igst_std': vd_wls_igst[~np.isnan(vd_wls_igst)].std(),
                    'Distlatlon_kf_avg': vd_kf.mean(), 'Distlatlon_kf_std': vd_kf.std(),
                    'Distlatlon_kf_igst_avg': vd_kf_igst.mean(), 'Distlatlon_kf_igst_std': vd_kf_igst.std(),
                    'score_bl': score_bl, 'score_wls': score_wls, 'score_kf': score_kf,
                    'Distecef_bl_avg': decef_bl.mean(), 'Distecef_bl_std': decef_bl.std(),
                    'Distecef_wls_avg': Distecef_wls_avg, 'Distecef_wls_std': Distecef_wls_std,
                    'Distecef_wls_igst_avg': decef_wls_igst[~np.isnan(decef_wls_igst)].mean(),
                    'Distecef_wls_igst_std': decef_wls_igst[~np.isnan(decef_wls_igst)].std(),
                    'Distecef_kf_avg': decef_kf.mean(), 'Distecef_kf_std': decef_kf.std(),
                    'Distecef_kf_igst_avg': decef_kf_igst.mean(), 'Distecef_kf_igst_std': decef_kf_igst.std(),
                }, index=[f'{i}'])
                record_dfs.append(record_df)
                # Write submission.csv
                records_df = pd.concat(record_dfs)
                # records_df.to_csv(savepath / 'records_scores_train_2022_ecef_igst.csv', index=False)

                print(
                    f'Baseline   : Distance in latlon from ground truth avg {vd_bl.mean():.3f}+{vd_bl.std():.3f} [m], Score {score_bl:.4f} [m], Distance in ECEF avg {decef_bl.mean():.3f}+{decef_bl.std():.3f} [m]')
                print(
                    f'Robust WLS : Distance in latlon from ground truth avg {Distlatlon_wls_avg:.3f}+{Distlatlon_wls_std:.3f} [m], Score {score_wls:.4f} [m], Distance in ECEF avg {Distecef_wls_avg:.3f}+{Distecef_wls_std:.3f} [m]')
                print(
                    f'KF         : Distance in latlon from ground truth avg {vd_kf.mean():.3f}+{vd_kf.std():.3f} [m], Score {score_kf:.4f} [m], Distance in ECEF avg {decef_kf.mean():.3f}+{decef_kf.std():.3f} [m]')
# %%

# %% md
