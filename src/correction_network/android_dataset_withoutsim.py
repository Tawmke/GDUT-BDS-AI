########################################################################
# Author(s):    Ashwin Kanhere, Shubh Gupta
# Date:         21 September 2021
# Desc:         Create PyTorch DataLoader for Android measurements
########################################################################
import sys, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.random import default_rng
import time

from src.gnss_lib.utils import datetime_to_tow
import src.gnss_lib.coordinates as coord

pd.options.mode.chained_assignment = None  # default='warn'


class Android_GNSS_Dataset(Dataset):
    def __init__(self, config, transforms=None):
        self.root = config['root']
        self.raw_dir = config['raw_data_dir']
        self.in_root = self.raw_dir #os.path.join(self.root, self.raw_dir)

        self.data_dir = config['data_dir']
        self.out_root = os.path.join(self.root, self.data_dir)
        if not os.path.exists(self.out_root):
            os.makedirs(self.out_root)

        # Config information for when files are opened and sampled by DataLoader
        self.max_open_files = config['max_open_files']  # cache size
        # self.guess_range = config['guess_range']
        self.history = config['history']
        self.nullaltitude2022list=config['nullaltitude2022list']
        self.zeroobs2022trajs=config['zeroobs2022trajs']
        self.rng = default_rng()
        if not config['bias_fname'] == 0:
            self.bias_pd = pd.read_csv(os.path.join(self.root, "android_info", config['bias_fname']))
            print("Bias loading: ", config['bias_fname'])
        else:
            self.bias_pd = None
            print("Not loading bias")

        self.transform = transforms

        # Config information for saving chunks
        self.chunk_size = config['chunk_size']
        try:
            self.max_sats = config['max_sats']
            scan_sats = False
        except KeyError:
            scan_sats = True
            self.max_sats = 0

        # Scan directory and store all trace/phone paths
        self.trace_paths = []
        for trace_name in sorted(os.listdir(self.in_root)):
            # ignoring trajs before 2021
            if '2021' in trace_name:
                # ignoring trajs without altitude
                if trace_name not in self.nullaltitude2022list:
                    _path = os.path.join(self.in_root, trace_name)
                    for phone_type in sorted(os.listdir(_path)):
                        traj_name=os.path.join(trace_name, phone_type)
                        #ignoring trajs with 0 obs
                        if traj_name not in self.zeroobs2022trajs:
                            self.trace_paths.append(traj_name)
        # record processing date
        filer = open(self.out_root + '/nan_records.txt', 'a+')
        filer.write(time.strftime("Start, %Y-%m-%d %H:%M:%S \n", time.localtime()))
        filer.close()

        filer = open(self.out_root + '/sat_num_records.txt', 'a+')
        filer.write(time.strftime("Start, %Y-%m-%d %H:%M:%S \n", time.localtime()))
        filer.close()
        # Scan files, create indices for loading, and store misc. data info
        self.num_traj = len(self.trace_paths)
        for _path in self.trace_paths:
            # Check if path, and files in path exist
            outpath = os.path.join(self.out_root, _path)
            fexists = False
            if os.path.exists(outpath):
                files = os.listdir(outpath)
                if files:
                    fexists = True
            # If files or path don't exist, process and chunk data
            if not fexists:
                print('Processed file not found. Chunking and saving data for ', _path)
                self.chunk_n_save(_path, save=True)
                print('Max. visible satellites: ', self.max_sats)
            # Even if the files exist, if max_number of sats is unknown,
            # scan the original dataset for this number
            elif scan_sats:
                print('Num of max sats unknown. Scanning data for ', _path)
                self.chunk_n_save(_path, save=False)
                print('Max. visible satellites: ', self.max_sats)

        print('Files scanned/saved. Creating lookup indices')
        # record processing end time
        filer = open(self.out_root + '/nan_records.txt', 'a+')
        filer.write(time.strftime("End, %Y-%m-%d %H:%M:%S \n", time.localtime()))
        filer.close()

        filer = open(self.out_root + '/sat_num_records.txt', 'a+')
        filer.write(time.strftime("End, %Y-%m-%d %H:%M:%S \n", time.localtime()))
        filer.close()
        # Add chunked files to index for loading later
        self.meas_file_paths = {}
        self.timestep_counts = {}
        for path_idx, _path in enumerate(self.trace_paths):
            for fpath in sorted(os.listdir(os.path.join(self.out_root, _path))):
                traj_num = os.path.split(fpath)[1].split(".")[0]
                traj_key = (_path, traj_num)
                if traj_key not in self.meas_file_paths.keys():
                    self.meas_file_paths[traj_key] = []
                self.meas_file_paths[traj_key] = os.path.join(self.out_root, _path, fpath)
                data = pd.read_csv(os.path.join(self.out_root, _path, fpath))
                self.timestep_counts[traj_key] = len(data['millisSinceGpsEpoch'].unique())

        self.full_counts = {key: self.timestep_counts[key]
                            for key in self.timestep_counts.keys()}
        self.N_total = sum(self.full_counts.values())

        indices = []
        keyList = sorted(self.full_counts.keys())
        traj_idx = 0
        timestep = 0
        for i in range(self.N_total):
            key = keyList[traj_idx]
            indices.append((key, timestep))
            timestep += 1
            if timestep >= self.timestep_counts[key]:
                timestep = 0
                traj_idx += 1
        self.indices = indices

    def get_files(self, key):
        if not hasattr(self, 'cache_traj'):
            self.cache_traj = dict()
            self.cache_times = dict()

        key_hash = str(key)
        if key_hash in self.cache_traj.keys():
            key_file = self.cache_traj[key_hash]
            times = self.cache_times[key_hash]
        else:
            key_file = pd.read_csv(self.meas_file_paths[key])
            times = key_file['millisSinceGpsEpoch'].unique()
            if len(self.cache_traj) >= self.max_open_files:
                pop_key = list(self.cache_traj.keys())[0]
                self.cache_traj.pop(pop_key)
                self.cache_times.pop(pop_key)
            self.cache_traj[key_hash] = key_file
            self.cache_times[key_hash] = times
        return key_file, times

    # def add_guess_noise(self, true_XYZb):
    #     guess_noise = np.array([
    #         self.rng.uniform(-self.guess_range[0], self.guess_range[0]),
    #         self.rng.uniform(-self.guess_range[1], self.guess_range[1]),
    #         self.rng.uniform(-self.guess_range[2], self.guess_range[2]),  # x, y, z
    #         self.rng.uniform(0, self.guess_range[3]),  # cdt
    #     ])
    #     return true_XYZb + guess_noise

    def __getitem__(self, idx):#, guess_XYZb=None
        key, timestep = self.indices[idx]
        key_file, times = self.get_files(key)
        try:
            data = key_file[key_file['millisSinceGpsEpoch'] == times[timestep]]
        except:
            raise ValueError('idx not found in dataset')
        _data0 = data.iloc[0]

        # Select random initialization
        true_XYZb = np.array([_data0['ecefX'], _data0['ecefY'], _data0['ecefZ'], _data0['b']])
        # if guess_XYZb is None:
        #     guess_XYZb = self.add_guess_noise(true_XYZb)  # Generate guess by adding noise to groundtruth
        #         guess_XYZb = np.copy(true_XYZb)         # 0 noise for debugging
        guess_XYZb= np.array([_data0['WlsPositionXEcefMeters'], _data0['WlsPositionYEcefMeters'],
                              _data0['WlsPositionZEcefMeters'], _data0['Wls_bias']])

        # Transform to NED frame
        ref_local = coord.LocalCoord.from_ecef(guess_XYZb[:3])
        guess_NEDb = np.copy(guess_XYZb)
        guess_NEDb[:3] = ref_local.ecef2ned(guess_XYZb[:3, None])[:, 0]  # position
        #         guess_NEDb[4:7] = ref_local.ecef2nedv(guess_XYZb[4:7, None])[:, 0]    # velocity

        true_NEDb = np.copy(true_XYZb)
        true_NEDb[:3] = ref_local.ecef2ned(true_XYZb[:3, None])[:, 0]  # position

        # Primary feature extraction
        # expected_pseudo, satXYZV = expected_measurements(data, guess_XYZb)
        expected_pseudo, satXYZV = expected_measurements_GSDC2022(data, guess_XYZb)
        residuals = (data['PrM'] - expected_pseudo).to_numpy()
        if self.bias_pd is not None:
            bias_slice = self.bias_pd[self.bias_pd['tracePath'] == key[0]]
            for idx in range(len(residuals)):
                svid = data["SvName"].values[idx]
                residuals[idx] = residuals[idx] - bias_slice.loc[bias_slice['SvName'] == svid, 'bias'].to_numpy()[0]
        los_vector = (satXYZV[['x', 'y', 'z']] - guess_XYZb[:3])
        los_vector = los_vector.div(np.sqrt(np.square(los_vector).sum(axis=1)), axis='rows').to_numpy()
        los_vector = ref_local.ecef2nedv(los_vector)

        features = np.concatenate((np.reshape(residuals, [-1, 1]), los_vector), axis=1)

        sample = {
            'features': torch.Tensor(features),
            'true_correction': (true_NEDb - guess_NEDb)[:3],
            'guess': guess_XYZb
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return int(self.N_total)

    def chunk_n_save(self, _path, save=True):
        # Load ground truth and append ECEF positions to dataframe
        gt_name = "ground_truth.csv"
        outpath = os.path.join(self.out_root, _path)
        gt = pd.read_csv(os.path.join(self.in_root, _path, gt_name))
        # lla = gt[['latDeg', 'lngDeg', 'heightAboveWgs84EllipsoidM']].to_numpy()
        lla = gt[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy()
        # NOTE: Implementing height correction as mentioned in the dataset
        ## forget to chunk before 1027.1700
        # lla[:, 2] = lla[:, 2] - 61
        ecef = coord.geodetic2ecef(lla)
        gt['ecefX'] = ecef[:, 0]
        gt['ecefY'] = ecef[:, 1]
        gt['ecefZ'] = ecef[:, 2]
        gt['b'] = np.zeros_like(ecef[:, 0])

        names = os.listdir(os.path.join(self.in_root, _path))
        # renamed for GSDC2022
        for name in names:
            if name.endswith("_gnss.csv"):
                og_name = name
        data = pd.read_csv(os.path.join(self.in_root, _path, og_name))
        # data["PrM"] = data["rawPrM"] \
        #               + data["satClkBiasM"] \
        #               - data["isrbM"] \
        #               - data["ionoDelayM"] \
        #               - data["tropoDelayM"]
        data["PrM"] = data["RawPseudorangeMeters"] \
                      + data["SvClockBiasMeters"] \
                      - data["IsrbMeters"] \
                      - data["IonosphericDelayMeters"] \
                      - data["TroposphericDelayMeters"]
        data["SvName"] = data["SignalType"] + "_" + str(data["Svid"])
        data["SvName"] = data["ConstellationType"].replace(
            [1, 3, 4, 5, 6], ["G", "R", "Q", "B", "E"]) \
                         + data["Svid"].astype(str)
        derived_timestamps = data['utcTimeMillis'].unique()
        gt_timestamps = gt['UnixTimeMillis'].unique()
        gt["millisSinceGpsEpoch"]=gt['UnixTimeMillis']

        # The timestamps in derived are one epoch ahead. We need to map each epoch
        # in derived to the prior one (in Raw).
        indexes = np.searchsorted(gt_timestamps, derived_timestamps)
        # from_t_to_fix_derived = dict(zip(derived_timestamps, gt_timestamps[indexes - 1])) # -1 because GSDC2021 mismatch
        from_t_to_fix_derived = dict(zip(derived_timestamps, gt_timestamps[indexes]))
        data['millisSinceGpsEpoch_old'] = data['utcTimeMillis'].values
        data['millisSinceGpsEpoch'] = np.array(
            list(map(lambda v: from_t_to_fix_derived[v], data['utcTimeMillis'])))

        count = 0
        chunk = []
        chunk_num = 0
        null_break=False
        min_sat_all=1e6
        max_sat_all=0
        min_sat_gpsl1=1e6
        max_sat_gpsl1=0
        min_sat_glog1=1e6
        max_sat_glog1=0
        min_sat_bdsb1i=1e6
        max_sat_bdsb1i=0
        min_sat_gale1=1e6
        max_sat_gale1=0
        min_sat_gpsl5=1e6
        max_sat_gpsl5=0
        min_sat_gale5a=1e6
        max_sat_gale5a=0
        min_sat_qzsj1=1e6
        max_sat_qzsj1=0
        min_sat_qzsj5=1e6
        max_sat_qzsj5=0

        for idx, dframe in data.groupby("millisSinceGpsEpoch"):
            # Drop duplicate data if it exists
            dframe.drop_duplicates(inplace=True)
            # remove nans in dframe 20221025
            dframe = dframe[dframe['SignalType'].notna()]
            # Remove all signals except GPS L1
            dframe_L1 = dframe.drop(index=dframe.loc[dframe["SignalType"] != "GPS_L1"].index)
            dframe_L1.drop_duplicates(subset=['Svid'], inplace=True)
            dframe_L1.reset_index(drop=True, inplace=True)
            # Use satellite measurements to obtain receiver clock ground truth estimate
            gt_row = gt["millisSinceGpsEpoch"] == idx
            gt_slice = gt.loc[gt_row]

            # record sat nums
            if len(dframe)>max_sat_all:
                max_sat_all=len(dframe)
            if len(dframe)<min_sat_all:
                min_sat_all=len(dframe)
            if len(dframe_L1)>max_sat_gpsl1:
                max_sat_gpsl1=len(dframe_L1)
            if len(dframe_L1)<min_sat_gpsl1:
                min_sat_gpsl1=len(dframe_L1)
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "GLO_G1"].index))
            if tmpnum>max_sat_glog1:
                max_sat_glog1=tmpnum
            if tmpnum<min_sat_glog1:
                min_sat_glog1=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "BDS_B1I"].index))
            if tmpnum>max_sat_bdsb1i:
                max_sat_bdsb1i=tmpnum
            if tmpnum<min_sat_bdsb1i:
                min_sat_bdsb1i=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "GAL_E1"].index))
            if tmpnum>max_sat_gale1:
                max_sat_gale1=tmpnum
            if tmpnum<min_sat_gale1:
                min_sat_gale1=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "GPS_L5"].index))
            if tmpnum>max_sat_gpsl5:
                max_sat_gpsl5=tmpnum
            if tmpnum<min_sat_gpsl5:
                min_sat_gpsl5=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "GAL_E5A"].index))
            if tmpnum>max_sat_gale5a:
                max_sat_gale5a=tmpnum
            if tmpnum<min_sat_gale5a:
                min_sat_gale5a=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "QZS_J1"].index))
            if tmpnum>max_sat_qzsj1:
                max_sat_qzsj1=tmpnum
            if tmpnum<min_sat_qzsj1:
                min_sat_qzsj1=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "QZS_J5"].index))
            if tmpnum>max_sat_qzsj5:
                max_sat_qzsj5=tmpnum
            if tmpnum<min_sat_qzsj5:
                min_sat_qzsj5=tmpnum
            # if (idx>=1619481583000 and idx <=1619481600000) or (chunk_num>=130):
            #     checknow=1
            # In case gt_slice is empty, find closest ground truth
            if len(gt_slice) == 0:
                low_gt_half = gt.loc[gt['UnixTimeMillis'] <= idx]
                try:
                    low_diff = np.abs(low_gt_half.iloc[-1]['millisSinceGpsEpoch'] - idx)
                except IndexError:
                    low_diff = 10000
                high_gt_half = gt.loc[gt['millisSinceGpsEpoch'] >= idx]
                try:
                    high_diff = np.abs(high_gt_half.iloc[0]['millisSinceGpsEpoch'] - idx)
                except IndexError:
                    high_diff = 10000
                if low_diff < high_diff:
                    gt_slice = low_gt_half.iloc[[-1]]
                else:
                    gt_slice = high_gt_half.iloc[[0]]
            # gt_slice['b'] = solve_gt_b(dframe, gt_slice)
            gt_slice['b'] = solve_gt_b_GSDC2022(dframe, gt_slice)
            if len(dframe_L1)>0:
                gt_slice['Wls_bias']=solve_wls_b_GSDC2022(dframe_L1, gt_slice)
            else:
                gt_slice['Wls_bias'] = 0
            # Update maximum number of visible sats
            if len(dframe_L1) > self.max_sats:
                self.max_sats = len(dframe_L1)
            # Add ground truth to the measurement data frame
            shaped_ones = np.ones(len(dframe_L1))
            # .to_numpy() required because gt_slice is always a DataFrame (ensures receiver bias value is always a scalar)
            dframe_L1.loc[:, 'ecefX'] = gt_slice['ecefX'].to_numpy() * shaped_ones
            dframe_L1.loc[:, 'ecefY'] = gt_slice['ecefY'].to_numpy() * shaped_ones
            dframe_L1.loc[:, 'ecefZ'] = gt_slice['ecefZ'].to_numpy() * shaped_ones
            dframe_L1.loc[:, 'b'] = gt_slice['b'].to_numpy() * shaped_ones
            dframe_L1.loc[:, 'Wls_bias'] = gt_slice['Wls_bias'].to_numpy() * shaped_ones
            if dframe_L1.isnull().values.any():
                nan_record=pd.DataFrame({'col':dframe_L1.isnull().any()})
                nan_names=nan_record[nan_record.col==True].index.to_list()
                if 'ecefX' in nan_names or 'b' in nan_names or 'Wls_bias' in nan_names:
                    print(dframe_L1)
                    filer=open(self.out_root+'/nan_records.txt','a+')
                    print(f'NaNs in DF columns {nan_names} at one epoch, sat num {len(dframe)}')
                    filer.write(f'{_path} NaNs in DF columns {nan_names} at one epoch, sat num {len(dframe)}, visible GPS L1 sat num {len(dframe_L1)}\n')
                    filer.close()
                    null_break=True
                    break
                    # raise ValueError('NaNs in DF at one epoch')
            chunk.append(dframe_L1)
            count += 1
            # Save newly created chunk
            if count == self.chunk_size:
                chunk_df = pd.concat(chunk)
                #                 fname = og_name.split(".")[0]+"_"+str(chunk_num) +".csv"
                fname = str(chunk_num) + ".csv"
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                if save and (not null_break):
                    if chunk_df.isnull().values.any():
                        nan_record=pd.DataFrame({'col':chunk_df.isnull().any()})
                        nan_names=nan_record[nan_record.col==True].index.to_list()
                        if 'ecefX' in nan_names or 'b' in nan_names or 'Wls_bias' in nan_names:
                            print(chunk_df)
                            raise ValueError('NaNs in chunked DF')
                    chunk_df.to_csv(os.path.join(outpath, fname))
                count = 0
                chunk_num += 1
                chunk = []
        # recording sat nums
        filer = open(self.out_root + '/sat_num_records.txt', 'a+')
        filer.write(f'{_path}, satnum, {min_sat_all}, {max_sat_all}, GPS_L1, {min_sat_gpsl1}, {max_sat_gpsl1}, '
                    f'GLO_G1, {min_sat_glog1}, {max_sat_glog1}, BDS_B1I, {min_sat_bdsb1i}, {max_sat_bdsb1i}, '
                    f'GAL_E1, {min_sat_gale1}, {max_sat_gale1}, GPS_L5, {min_sat_gpsl5}, {max_sat_gpsl5}, '
                    f'GAL_E5A, {min_sat_gale5a}, {max_sat_gale5a}, QZS_J1, {min_sat_qzsj1}, {max_sat_qzsj1}, '
                    f'QZS_J5, {min_sat_qzsj5}, {max_sat_qzsj5}\n')
        filer.close()

        # save left steps below chunk size for saving 1025
        if count>0:
            chunk_df = pd.concat(chunk)
            #         fname = og_name.split(".")[0]+"_"+str(chunk_num) +".csv"
            fname = str(chunk_num) + ".csv"
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            if save and (not null_break):
                if chunk_df.isnull().values.any():
                    nan_record=pd.DataFrame({'col':chunk_df.isnull().any()})
                    nan_names=nan_record[nan_record.col==True].index.to_list()
                    if 'ecefX' in nan_names or 'b' in nan_names or 'Wls_bias' in nan_names:
                        print(chunk_df)
                        raise ValueError('NaNs in chunked DF')
                chunk_df.to_csv(os.path.join(outpath, fname))


def solve_gt_b(dframe, gt_slice, max_iter=20, tol=1e-3):
    gt_ecef = np.zeros([1, 3])
    gt_ecef[0, 0] = gt_slice["ecefX"]
    gt_ecef[0, 1] = gt_slice["ecefY"]
    gt_ecef[0, 2] = gt_slice["ecefZ"]

    sat_pos = np.hstack((dframe["xSatPosM"].to_numpy().reshape(-1, 1),
                         dframe["ySatPosM"].to_numpy().reshape(-1, 1),
                         dframe["zSatPosM"].to_numpy().reshape(-1, 1)))
    gt_pos = np.tile(gt_ecef, (sat_pos.shape[0], 1))

    gt_ranges = np.linalg.norm(gt_pos - sat_pos, axis=1)

    G = np.ones((sat_pos.shape[0], 1))
    #     G[:,:3] = np.divide(gt_pos - sat_pos,gt_ranges.reshape(-1,1))

    W = np.diag(1. / dframe["rawPrUncM"] ** 2)
    rb = 0.
    for _ in range(max_iter):
        rho_diff = dframe["PrM"].to_numpy() \
                   - gt_ranges \
                   - rb

        rho_diff = rho_diff.reshape(-1, 1)

        delta = np.linalg.pinv(W.dot(G)).dot(W).dot(rho_diff)
        rb_new = rb + delta
        rb = rb_new
        if np.abs(delta) < tol:
            break
    return rb_new

def solve_gt_b_GSDC2022(dframe, gt_slice, max_iter=20, tol=1e-3):
    gt_ecef = np.zeros([1, 3])
    gt_ecef[0, 0] = gt_slice["ecefX"]
    gt_ecef[0, 1] = gt_slice["ecefY"]
    gt_ecef[0, 2] = gt_slice["ecefZ"]

    sat_pos = np.hstack((dframe["SvPositionXEcefMeters"].to_numpy().reshape(-1, 1),
                         dframe["SvPositionYEcefMeters"].to_numpy().reshape(-1, 1),
                         dframe["SvPositionZEcefMeters"].to_numpy().reshape(-1, 1)))
    gt_pos = np.tile(gt_ecef, (sat_pos.shape[0], 1))

    gt_ranges = np.linalg.norm(gt_pos - sat_pos, axis=1)

    G = np.ones((sat_pos.shape[0], 1))
    #     G[:,:3] = np.divide(gt_pos - sat_pos,gt_ranges.reshape(-1,1))

    W = np.diag(1. / dframe["RawPseudorangeUncertaintyMeters"] ** 2)
    rb = 0.
    for _ in range(max_iter):
        rho_diff = dframe["PrM"].to_numpy() \
                   - gt_ranges \
                   - rb

        rho_diff = rho_diff.reshape(-1, 1)

        delta = np.linalg.pinv(W.dot(G)).dot(W).dot(rho_diff)
        rb_new = rb + delta
        rb = rb_new
        if np.abs(delta) < tol:
            break
    return rb_new

def solve_wls_b_GSDC2022(dframe, gt_slice, max_iter=20, tol=1e-3):
    gt_ecef = np.zeros([1, 3])
    gt_ecef[0, 0] = dframe["WlsPositionXEcefMeters"][0]
    gt_ecef[0, 1] = dframe["WlsPositionYEcefMeters"][0]
    gt_ecef[0, 2] = dframe["WlsPositionZEcefMeters"][0]

    sat_pos = np.hstack((dframe["SvPositionXEcefMeters"].to_numpy().reshape(-1, 1),
                         dframe["SvPositionYEcefMeters"].to_numpy().reshape(-1, 1),
                         dframe["SvPositionZEcefMeters"].to_numpy().reshape(-1, 1)))
    gt_pos = np.tile(gt_ecef, (sat_pos.shape[0], 1))

    gt_ranges = np.linalg.norm(gt_pos - sat_pos, axis=1)

    G = np.ones((sat_pos.shape[0], 1))
    #     G[:,:3] = np.divide(gt_pos - sat_pos,gt_ranges.reshape(-1,1))

    W = np.diag(1. / dframe["RawPseudorangeUncertaintyMeters"] ** 2)
    rb = 0.
    for _ in range(max_iter):
        rho_diff = dframe["PrM"].to_numpy() \
                   - gt_ranges \
                   - rb

        rho_diff = rho_diff.reshape(-1, 1)

        delta = np.linalg.pinv(W.dot(G)).dot(W).dot(rho_diff)
        rb_new = rb + delta
        rb = rb_new
        if np.abs(delta) < tol:
            break
    return rb_new

def expected_measurements(dframe, guess_XYZb):
    satX = dframe.loc[:, "xSatPosM"].to_numpy()
    satY = dframe.loc[:, "ySatPosM"].to_numpy()
    satZ = dframe.loc[:, "zSatPosM"].to_numpy()
    satvX = dframe.loc[:, "xSatVelMps"].to_numpy()
    satvY = dframe.loc[:, "ySatVelMps"].to_numpy()
    satvZ = dframe.loc[:, "zSatVelMps"].to_numpy()
    gt_ranges = np.sqrt((satX - guess_XYZb[0]) ** 2 \
                        + (satY - guess_XYZb[1]) ** 2 \
                        + (satZ - guess_XYZb[2]) ** 2)
    #     gt_ranges = gt_ranges.values
    expected_rho = gt_ranges + guess_XYZb[3]
    satXYZV = pd.DataFrame()
    satXYZV['x'] = satX
    satXYZV['y'] = satY
    satXYZV['z'] = satZ
    satXYZV['vx'] = satvX
    satXYZV['vy'] = satvY
    satXYZV['vz'] = satvZ
    return expected_rho, satXYZV

def expected_measurements_GSDC2022(dframe, guess_XYZb):
    satX = dframe.loc[:, "SvPositionXEcefMeters"].to_numpy()
    satY = dframe.loc[:, "SvPositionYEcefMeters"].to_numpy()
    satZ = dframe.loc[:, "SvPositionZEcefMeters"].to_numpy()
    satvX = dframe.loc[:, "SvVelocityXEcefMetersPerSecond"].to_numpy()
    satvY = dframe.loc[:, "SvVelocityYEcefMetersPerSecond"].to_numpy()
    satvZ = dframe.loc[:, "SvVelocityZEcefMetersPerSecond"].to_numpy()
    gt_ranges = np.sqrt((satX - guess_XYZb[0]) ** 2 \
                        + (satY - guess_XYZb[1]) ** 2 \
                        + (satZ - guess_XYZb[2]) ** 2)
    #     gt_ranges = gt_ranges.values
    expected_rho = gt_ranges + guess_XYZb[3]
    satXYZV = pd.DataFrame()
    satXYZV['x'] = satX
    satXYZV['y'] = satY
    satXYZV['z'] = satZ
    satXYZV['vx'] = satvX
    satXYZV['vy'] = satvY
    satXYZV['vz'] = satvZ
    return expected_rho, satXYZV