import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.tools import convert_tsf_to_dataframe
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


class CarbonCaptureDataset(Dataset):
    def __init__(
        self,
        root_path="/data/tsfm23/datasets/exogs_expts/carbon_data_processed/",
        data_path="official__20220210_smooth_window_16.csv",
        size=None,
        flag="train",
        scale=True,
        return_dict=False,
        percent=None,
        forecast_channel_indices=[6, 7],
        features="M",
        time_col_name="time",
        target="OT",
        use_time_features=False,
        timeenc=0,
        freq="s",
        max_len=-1,
        train_all=False,
    ):
        if size == None:
            raise ("size must be provided")
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.flag = flag
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.return_dict = return_dict
        self.percent = percent

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features
        self.forecast_channel_indices = forecast_channel_indices

        self.__read_data__()

        # GPT4TS uses univariate data. The following needed for that.
        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            len(df_raw) // 2 - self.seq_len,
            len(df_raw) // 2 + len(df_raw) // 4 - self.seq_len,
        ]
        border2s = [len(df_raw) // 2, len(df_raw) // 2 + len(df_raw) // 4, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0 and self.percent is not None:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[
                1:
            ]  # this is NOT selecting the index and timestamp col
            self.df_data = df_raw[cols_data]
        elif self.features == "S":
            self.df_data = df_raw[[self.target]]

        self.df_data = df_raw[cols_data]
        self.num_channels = len(self.df_data.columns)

        if self.scale:
            train_data = self.df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            self.data = self.scaler.transform(self.df_data.values)
        else:
            self.data = self.df_data.values

        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp[self.time_col_name].apply(
                lambda row: row.month, 1
            )
            df_stamp["day"] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp[self.time_col_name].apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp[self.time_col_name].apply(
                lambda row: row.hour, 1
            )
            data_stamp = df_stamp.drop([self.time_col_name], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = self.data[border1:border2, self.forecast_channel_indices]
        self.data_y = self.data[border1:border2, self.forecast_channel_indices]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id : feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id : feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class BikeSharingDataset(Dataset):
    def __init__(
        self,
        root_path="/data/tsfm23/datasets/exogs_expts/bike_sharing_dataset/processed_data",
        data_path="bike_sharing_hourly_processed.csv",
        size=None,
        flag="train",
        scale=True,
        return_dict=False,
        percent=None,
        forecast_channel_indices=[11, 12, 13],
        features="M",
        time_col_name="timestamp",
        target="OT",
        use_time_features=False,
        timeenc=0,
        freq="s",
        max_len=-1,
        train_all=False,
    ):
        if size == None:
            raise ("size must be provided")
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.flag = flag
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.return_dict = return_dict
        self.percent = percent

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features
        self.forecast_channel_indices = forecast_channel_indices

        self.__read_data__()

        # GPT4TS uses univariate data. The following needed for that.
        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            len(df_raw) // 2 - self.seq_len,
            len(df_raw) // 2 + len(df_raw) // 4 - self.seq_len,
        ]
        border2s = [len(df_raw) // 2, len(df_raw) // 2 + len(df_raw) // 4, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0 and self.percent is not None:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[
                2:
            ]  # this is NOT selecting the index and timestamp col
            self.df_data = df_raw[cols_data]
        elif self.features == "S":
            self.df_data = df_raw[[self.target]]
        self.num_channels = len(self.df_data)

        if self.scale:
            train_data = self.df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            self.data = self.scaler.transform(self.df_data.values)
        else:
            self.data = self.df_data.values

        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp[self.time_col_name].apply(
                lambda row: row.month, 1
            )
            df_stamp["day"] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp[self.time_col_name].apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp[self.time_col_name].apply(
                lambda row: row.hour, 1
            )
            data_stamp = df_stamp.drop([self.time_col_name], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = self.data[border1:border2, self.forecast_channel_indices]
        self.data_y = self.data[border1:border2, self.forecast_channel_indices]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id : feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id : feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_IT_service(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="Service.csv",
        time_col_name="date",
        target="OT",
        scale=True,
        use_time_features=False,
        timeenc=0,
        freq="s",
        train_split=0.6,
        test_split=0.2,
        forecast_channel_indices=[
            13,
            40,
            44,
            47,
            50,
            54,
            57,
            61,
            64,
            67,
            70,
            74,
            78,
            82,
            85,
            89,
            92,
            95,
            99,
            102,
            105,
            14,
            41,
            45,
            48,
            51,
            55,
            58,
            62,
            65,
            68,
            71,
            75,
            79,
            83,
            86,
            90,
            93,
            96,
            100,
            103,
            106,
            11,
            12,
            38,
            39,
            42,
            43,
            46,
            49,
            52,
            53,
            56,
            59,
            60,
            63,
            66,
            69,
            72,
            73,
            77,
            80,
            81,
            84,
            87,
            88,
            91,
            94,
            97,
            98,
            101,
            104,
        ],
        percent=100,
        max_len=-1,
        train_all=False,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            raise ("size must be provided")
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features
        self.forecast_channel_indices = forecast_channel_indices

        # train test ratio
        self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        # GPT4TS uses univariate data. The following needed for that.
        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        # cols.remove(self.target)
        # df_raw = df_raw[['date'] + cols ]
        # print(cols)

        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp[self.time_col_name].apply(
                lambda row: row.month, 1
            )
            df_stamp["day"] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp[self.time_col_name].apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp[self.time_col_name].apply(
                lambda row: row.hour, 1
            )
            data_stamp = df_stamp.drop([self.time_col_name], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2, self.forecast_channel_indices]
        self.data_y = data[border1:border2, self.forecast_channel_indices]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id : feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id : feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_IT_application(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="Application.csv",
        time_col_name="date",
        target="OT",
        scale=True,
        use_time_features=False,
        timeenc=0,
        freq="s",
        train_split=0.6,
        test_split=0.2,
        forecast_channel_indices=[0, 1, 2, 3],
        percent=100,
        max_len=-1,
        train_all=False,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            raise ("size must be provided")
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features
        self.forecast_channel_indices = forecast_channel_indices

        # train test ratio
        self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        # GPT4TS uses univariate data. The following needed for that.
        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        # cols.remove(self.target)
        # df_raw = df_raw[['date'] + cols ]
        # print(cols)
        """
        There are 39 columns in IT application data 
        and 8834 observation 
        if take 0.7 as training data then there are
        6183 observation in train data
        1766 -||- in test data 
        885 -||- in valiation
        according to the set_type it selects the length of the data using boarders
        if set_type 0 it takes train data from  0:num_train
        if set_typr 1 it takes val data from (num_tran-seq_len): (num_train+num_val)
        and so on 
        """
        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp[self.time_col_name].apply(
                lambda row: row.month, 1
            )
            df_stamp["day"] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp[self.time_col_name].apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp[self.time_col_name].apply(
                lambda row: row.hour, 1
            )
            data_stamp = df_stamp.drop([self.time_col_name], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2, self.forecast_channel_indices]
        self.data_y = data[border1:border2, self.forecast_channel_indices]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id : feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id : feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        percent=100,
        max_len=-1,
        train_all=False,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        print("self.data_x pruned = {}".format(self.data_x.shape))
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id : feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id : feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        percent=100,
        max_len=-1,
        train_all=False,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id : feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id : feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        percent=10,
        max_len=-1,
        train_all=False,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id : feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id : feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(
        self,
        root_path,
        flag="pred",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="15min",
        cols=None,
        percent=None,
        train_all=False,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["pred"]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[["date"]][border1:border2]
        tmp_stamp["date"] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq
        )

        df_stamp = pd.DataFrame(columns=["date"])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin : r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin : r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_TSF(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path=None,
        target="OT",
        scale=True,
        timeenc=0,
        freq="Daily",
        percent=10,
        max_len=-1,
        train_all=False,
    ):
        self.train_all = train_all

        self.seq_len = size[0]
        self.pred_len = size[2]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.max_len = max_len
        if self.max_len == -1:
            self.max_len = 1e8

        self.root_path = root_path
        self.data_path = data_path
        self.timeseries = self.__read_data__()

    def __read_data__(self):
        (
            df,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        ) = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        self.freq = frequency

        def dropna(x):
            return x[~np.isnan(x)]

        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]

        self.tot_len = 0
        self.len_seq = []
        self.seq_id = []
        for i in range(len(timeseries)):
            res_len = max(self.pred_len + self.seq_len - timeseries[i].shape[0], 0)
            pad_zeros = np.zeros(res_len)
            timeseries[i] = np.hstack([pad_zeros, timeseries[i]])

            _len = timeseries[i].shape[0]
            train_len = _len - self.pred_len
            if self.train_all:
                border1s = [0, 0, train_len - self.seq_len]
                border2s = [train_len, train_len, _len]
            else:
                border1s = [
                    0,
                    train_len - self.seq_len - self.pred_len,
                    train_len - self.seq_len,
                ]
                border2s = [train_len - self.pred_len, train_len, _len]
            border2s[0] = (
                border2s[0] - self.seq_len
            ) * self.percent // 100 + self.seq_len
            # print("_len = {}".format(_len))

            curr_len = (
                border2s[self.set_type]
                - max(border1s[self.set_type], 0)
                - self.pred_len
                - self.seq_len
                + 1
            )
            curr_len = max(0, curr_len)

            self.len_seq.append(np.zeros(curr_len) + self.tot_len)
            self.seq_id.append(np.zeros(curr_len) + i)
            self.tot_len += curr_len

        self.len_seq = np.hstack(self.len_seq)
        self.seq_id = np.hstack(self.seq_id)

        return timeseries

    def __getitem__(self, index):
        len_seq = self.len_seq[index]
        seq_id = int(self.seq_id[index])
        index = index - int(len_seq)

        _len = self.timeseries[seq_id].shape[0]
        train_len = _len - self.pred_len
        if self.train_all:
            border1s = [0, 0, train_len - self.seq_len]
            border2s = [train_len, train_len, _len]
        else:
            border1s = [
                0,
                train_len - self.seq_len - self.pred_len,
                train_len - self.seq_len,
            ]
            border2s = [train_len - self.pred_len, train_len, _len]
        border2s[0] = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len

        s_begin = index + border1s[self.set_type]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        if self.set_type == 2:
            s_end = -self.pred_len

        data_x = self.timeseries[seq_id][s_begin:s_end]
        data_y = self.timeseries[seq_id][r_begin:r_end]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)
        # if self.set_type == 2:
        #     print("data_x.shape = {}, data_y.shape = {}".format(data_x.shape, data_y.shape))

        return data_x, data_y, data_x, data_y

    def __len__(self):
        if self.set_type == 0:
            # return self.tot_len
            return min(self.max_len, self.tot_len)
        else:
            return self.tot_len


if __name__ == "__main__":
    dset = Dataset_IT_application(
        root_path="/data/dnn_forecasting/FM/ijcai24_target/data/bizitops",
        size=[96, 0, 24],
    )
    x, y, _, _ = dset.__getitem__(0)
    breakpoint()

    dset = CarbonCaptureDataset(
        size=[96, 0, 24],
    )
    x, y, _, _ = dset.__getitem__(0)
    breakpoint()

    dset = Dataset_IT_service(
        root_path="/data/dnn_forecasting/FM/ijcai24_target/data/bizitops",
        size=[96, 0, 24],
    )
    x, y, _, _ = dset.__getitem__(0)

    dset = BikeSharingDataset(
        size=[512, 0, 96],
    )
    x, y, _, _ = dset.__getitem__(0)
    breakpoint()
