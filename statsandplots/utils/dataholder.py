import numpy as np
import pandas as pd


class DataHolder:

    def __init__(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    data[key] = np.array(value)
        self.data = data
        self._container_type = self._get_container_type()

    def __contains__(self, item):
        if self._container_type == "pandas" or self._container_type == "dict":
            return item in self.data
        elif self._container_type == "numpy":
            return item < self.data.shape[1]

    def _get_container_type(self):
        if isinstance(self.data, (pd.Series, pd.DataFrame)):
            return "pandas"
        elif isinstance(self.data, np.ndarray):
            return "numpy"
        elif isinstance(self.data, dict):
            return "dict"
        else:
            raise ValueError(
                "Only numpy arrays or pandas dataframes/series are accepted."
            )

    def _numpy_index(self, index):
        if isinstance(index, tuple):
            return self.data[index]
        elif isinstance(index, int):
            return self.data[:, index]

    def _pandas_index(self, index):
        if isinstance(index, tuple):
            if pd.api.types.is_bool_dtype(index[0].dtype):
                return self.data.loc[index[0], index[1]]
            else:
                return self.data.iloc[index[0], self.data.columns.get_loc(index[1])]
        elif isinstance(index, str):
            return self.data[index]

    def _dict_index(self, index):
        if isinstance(index, tuple):
            return self.data[index[1]][index[0]]
        elif isinstance(index, str):
            return self.data[index]

    def __getitem__(self, index):
        if self._container_type == "numpy":
            return self._numpy_index(index)
        elif self._container_type == "pandas":
            return self._pandas_index(index)
        elif self._container_type == "dict":
            return self._dict_index(self, index)

    def min(self, index):
        return self.__getitem__(index).min()

    def max(self, index):
        return self.__getitem__(index).max()

    @property
    def size(self):
        if self._container_type == "numpy" or self._container_type == "pandas":
            return self.data.size
        elif self._container_type == "dict":
            return len(next(iter(self.data.values()))) * len(self.data.keys())

    @property
    def shape(self):
        if self._container_type == "numpy" or self._container_type == "pandas":
            return self.data.shape
        else:
            return (len(next(iter(self.data.values()))), len(self.data.keys()))

    def __len__(self):
        return self.size
