import pickle
import time

import numpy as np
import scipy.sparse as sparse


class TextPreprocess:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, data) -> object:
        start_time = time.time()
        _out = self.pipeline.fit(data)
        print(
            f"TextPreprocess.fit {data.name} {(time.time() - start_time):.2f} seconds"
        )
        return _out

    def fit_transform(self, data) -> object:
        start_time = time.time()
        _out = self.pipeline.fit_transform(data)
        print(
            f"TextPreprocess.fit {data.name} {(time.time() - start_time):.2f} seconds"
        )
        return self.pipeline.fit_transform(data)

    def transform(self, data) -> np.array:
        start_time = time.time()
        _out = self.pipeline.transform(data)
        print(
            f"TextPreprocess.transform {data.name} {(time.time() - start_time):.2f} seconds"
        )

        return _out

    def save(self, data, prefix_filename):
        start_time = time.time()
        data_sparse = sparse.csr_matrix(data)
        file_name = f"{prefix_filename}_{self.pipeline.name}.pkl"
        with open(file_name, "wb") as fp:
            pickle.dump(data_sparse, fp)
        print(
            f"TextPreprocess.save {file_name} {(time.time() - start_time):.2f} seconds"
        )
        return file_name

    def get_voc(self):
        voc = self.pipeline.get_voc()
        print(f"TextPreprocess.save_voc size {len(voc)}")
        return voc

    def save_voc(self, prefix_filename):
        voc = self.get_voc()
        file_name = f"{prefix_filename}_{self.pipeline.name}.pkl"
        with open(file_name, "wb") as fp:
            pickle.dump(voc, fp)
        print(f"TextPreprocess.save_voc {file_name}")
        return file_name


class TPWithExtraPreproc(TextPreprocess):
    """Example:
    bow_preproc = TPWithExtraPreproc(TfidfV1())
    X_train = bow_preproc.extra_preproc(X_train)
    bow_preproc.fit(X_train)
    ...
    """

    def __init__(self):
        TextPreprocess.__init__(self)

    def extra_preproc(self, data):
        # add some extra preproc
        return data
