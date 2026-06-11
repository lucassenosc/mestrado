
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import time
import random
import gc
import numpy as np
import joblib


_STATE_KEY = None
_MODEL = None
_X_TEST = None
_ADV = None
_TRAIN_TIME = 0.0


class FastCountingWisard:

    # Standard WiSARD exata com counting RAMs e bleaching eficiente.

    # Memória:
    #     table[ram][class] = (keys_sorted, counts)

    # Treino:
    #     count[class][ram][address] += 1

    # Inferência:
    #     score_class(b) = soma(count >= b para todas as RAMs)

    # Bleaching:
    #     em vez de testar b = 1, 2, 3, ..., max_count,
    #     testa apenas thresholds onde algum score pode mudar:
    #         b = count + 1


    def __init__(
        self,
        address_size,
        class_order,
        seed=42,
        bleach_max=None,
        predict_batch_size=1024
    ):
        self.address_size = int(address_size)
        self.class_order = [str(c) for c in class_order]
        self.class_values = np.array([int(c) for c in self.class_order], dtype=np.int32)
        self.n_classes = len(self.class_order)

        self.seed = int(seed)
        self.bleach_max = bleach_max
        self.predict_batch_size = int(predict_batch_size)

        self.mapping = None
        self.powers = np.left_shift(
            np.uint64(1),
            np.arange(self.address_size, dtype=np.uint64)
        ).astype(np.uint64)

        self.n_bits = None
        self.n_rams = None
        self.tables = None

    def _build_mapping(self, n_bits):
        self.n_bits = int(n_bits)
        self.n_rams = int(np.ceil(self.n_bits / self.address_size))

        rng = np.random.default_rng(self.seed)
        indices = np.arange(self.n_bits, dtype=np.int64)
        rng.shuffle(indices)

        total_needed = self.n_rams * self.address_size
        pad = total_needed - self.n_bits

        if pad > 0:
            extra = indices[:pad]
            indices = np.concatenate([indices, extra])

        self.mapping = indices.reshape(self.n_rams, self.address_size)

    def _addresses_for_ram(self, X, ram_idx):
        idx = self.mapping[int(ram_idx)]
        bits = np.asarray(X[:, idx], dtype=np.uint64)
        return bits @ self.powers

    def train(self, X_train, y_train):
        y_train = np.asarray(y_train, dtype=np.int32)

        if self.mapping is None:
            self._build_mapping(X_train.shape[1])

        masks = [(y_train == cls_value) for cls_value in self.class_values]

        tables = []

        for ram_idx in range(self.n_rams):
            addrs = self._addresses_for_ram(X_train, ram_idx)
            ram_tables = []

            for class_idx in range(self.n_classes):
                class_addrs = addrs[masks[class_idx]]

                if class_addrs.size == 0:
                    keys = np.empty(0, dtype=np.uint64)
                    counts = np.empty(0, dtype=np.uint32)
                else:
                    keys, counts = np.unique(class_addrs, return_counts=True)
                    keys = keys.astype(np.uint64, copy=False)
                    counts = counts.astype(np.uint32, copy=False)

                ram_tables.append((keys, counts))

            tables.append(ram_tables)

            del addrs

            if ram_idx % 32 == 0:
                gc.collect()

        self.tables = tables
        gc.collect()

    def _lookup_counts_for_ram_class(self, keys, counts, addresses):
        if keys.size == 0:
            return np.zeros(addresses.shape[0], dtype=np.uint32)

        pos = np.searchsorted(keys, addresses)
        pos_clip = np.minimum(pos, keys.size - 1)

        valid = (pos < keys.size) & (keys[pos_clip] == addresses)

        out = np.zeros(addresses.shape[0], dtype=np.uint32)
        out[valid] = counts[pos_clip[valid]]

        return out

    def _count_tensor_for_batch(self, X_batch):
        n = int(X_batch.shape[0])

        count_tensor = np.zeros(
            (n, self.n_classes, self.n_rams),
            dtype=np.uint32
        )

        for ram_idx in range(self.n_rams):
            addresses = self._addresses_for_ram(X_batch, ram_idx)

            for class_idx in range(self.n_classes):
                keys, counts = self.tables[ram_idx][class_idx]
                count_tensor[:, class_idx, ram_idx] = self._lookup_counts_for_ram_class(
                    keys,
                    counts,
                    addresses
                )

            del addresses

        return count_tensor

    def _first_by_class_order(self, class_indices):
        class_indices = [int(i) for i in class_indices]
        return min(class_indices)

    def _predict_one_from_counts(self, counts_one):
        scores = np.count_nonzero(counts_one >= 1, axis=1)
        max_score = int(np.max(scores))

        tied = np.flatnonzero(scores == max_score)

        if tied.size == 1:
            return int(tied[0])

        tied_counts = counts_one[tied, :]

        candidate_thresholds = np.unique(
            tied_counts[tied_counts > 0].astype(np.uint64) + np.uint64(1)
        )

        candidate_thresholds = candidate_thresholds[candidate_thresholds >= 2]

        if self.bleach_max is not None:
            candidate_thresholds = candidate_thresholds[
                candidate_thresholds <= int(self.bleach_max)
            ]

        if candidate_thresholds.size == 0:
            return int(self._first_by_class_order(tied))

        for b in candidate_thresholds:
            scores_b = np.count_nonzero(tied_counts >= b, axis=1)
            max_score_b = int(np.max(scores_b))

            tied_b_local = np.flatnonzero(scores_b == max_score_b)

            if tied_b_local.size == 1:
                return int(tied[int(tied_b_local[0])])

        return int(self._first_by_class_order(tied))

    def _predict_batch(self, X_batch):
        count_tensor = self._count_tensor_for_batch(X_batch)

        preds = []

        for i in range(count_tensor.shape[0]):
            pred_idx = self._predict_one_from_counts(count_tensor[i])
            preds.append(self.class_order[pred_idx])

        del count_tensor
        gc.collect()

        return preds

    def predict(self, X):
        preds = []
        n = int(X.shape[0])
        batch_size = max(1, int(self.predict_batch_size))

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            preds.extend(self._predict_batch(X[start:end]))

        return preds


def _ensure_state(config):
    global _STATE_KEY, _MODEL, _X_TEST, _ADV, _TRAIN_TIME

    if _STATE_KEY == config['state_key'] and _MODEL is not None:
        return 0.0

    np.random.seed(int(config.get('seed', 42)))
    random.seed(int(config.get('seed', 42)))

    X_train = joblib.load(config['X_train_path'], mmap_mode='r')
    y_train = joblib.load(config['y_train_path'], mmap_mode='r')
    _X_TEST = joblib.load(config['X_test_path'], mmap_mode='r')

    _ADV = {
        name: joblib.load(path, mmap_mode='r')
        for name, path in config['adv_paths'].items()
    }

    t0 = time.perf_counter()

    model = FastCountingWisard(
        address_size=int(config['addr']),
        class_order=config['class_order'],
        seed=int(config.get('seed', 42)),
        bleach_max=config.get('bleach_max', None),
        predict_batch_size=int(config.get('predict_batch_size', 1024))
    )

    model.train(X_train, y_train)

    _TRAIN_TIME = time.perf_counter() - t0

    _MODEL = model
    _STATE_KEY = config['state_key']

    del X_train, y_train
    gc.collect()

    return _TRAIN_TIME


def evaluate_standard_range_worker(config, start, end):
    train_time = _ensure_state(config)

    start = int(start)
    end = int(end)
    pid = os.getpid()

    t1 = time.perf_counter()
    yp_clean = _MODEL.predict(_X_TEST[start:end])
    infer_time_clean = time.perf_counter() - t1

    yp_adv = {}
    infer_time_adv = {}

    for atk_name, adv_matrix in _ADV.items():
        t2 = time.perf_counter()
        yp_adv[atk_name] = _MODEL.predict(adv_matrix[start:end])
        infer_time_adv[atk_name] = time.perf_counter() - t2

    return {
        'pid': pid,
        'start': start,
        'end': end,
        'yp_clean': yp_clean,
        'yp_adv': yp_adv,
        'train_time': float(train_time),
        'infer_time_clean': float(infer_time_clean),
        'infer_time_adv': infer_time_adv,
        'n_rams': int(_MODEL.n_rams),
        'address_size': int(_MODEL.address_size),
        'backend': 'FastCountingWisard',
        'fast_exact_bleaching': True,
    }
