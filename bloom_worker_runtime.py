
import os

# Evita oversubscription dentro de cada worker.
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

if 'libs' not in sys.path:
    sys.path.append('libs')
from libs.wisard import WiSARD

_STATE_KEY = None
_MODEL = None
_X_TEST = None
_ADV = None
_PAD_BUFFER = None
_TRAIN_TIME = 0.0


def _ensure_state(config):
    global _STATE_KEY, _MODEL, _X_TEST, _ADV, _PAD_BUFFER, _TRAIN_TIME

    if _STATE_KEY == config['state_key'] and _MODEL is not None:
        return 0.0

    np.random.seed(int(config.get('seed', 42)))
    random.seed(int(config.get('seed', 42)))

    X_train = joblib.load(config['X_train_path'], mmap_mode='r')
    y_train = joblib.load(config['y_train_path'], mmap_mode='r')
    _X_TEST = joblib.load(config['X_test_path'], mmap_mode='r')
    _ADV = {name: joblib.load(path, mmap_mode='r') for name, path in config['adv_paths'].items()}

    t0 = time.perf_counter()
    model = WiSARD(
        int(config['num_inputs']),
        int(config['num_classes']),
        int(config['addr']),
        int(config['bloom']),
        int(config['h'])
    )

    # Treino sem .tolist(): cada worker percorre o memmap read-only.
    for i in range(len(X_train)):
        model.train(X_train[i], int(y_train[i]))

    _TRAIN_TIME = time.perf_counter() - t0
    _MODEL = model
    _STATE_KEY = config['state_key']

    # Buffer reutilizável por processo para evitar np.pad por amostra.
    padded_len = int(config['num_inputs']) + int(getattr(model, 'pad_zeros', 0))
    _PAD_BUFFER = np.zeros(padded_len, dtype=X_train.dtype)

    del X_train, y_train
    gc.collect()

    return _TRAIN_TIME


def _predict_dynamic_with_prob(chunk):
    global _MODEL, _PAD_BUFFER

    model = _MODEL
    pad_zeros = int(model.pad_zeros)
    input_order = model.input_order
    discriminators = model.discriminators
    num_classes = len(discriminators)

    n_samples = len(chunk)
    preds = np.empty(n_samples, dtype=np.int32)
    probs = np.empty((n_samples, num_classes), dtype=np.float32)

    for row_idx, sample in enumerate(chunk):
        if pad_zeros > 0:
            _PAD_BUFFER[:len(sample)] = sample
            _PAD_BUFFER[len(sample):] = 0
            xv_padded = _PAD_BUFFER[input_order]
        else:
            xv_padded = sample[input_order]

        b = 1
        last_tie = np.array([0], dtype=np.int32)
        first_responses = None

        # Bleaching exponencial: evita b=1,2,3,... quando há empates muito longos.
        while b <= 200:
            model.set_bleaching(b)

            responses = np.empty(num_classes, dtype=np.float32)
            for c_idx, disc in enumerate(discriminators):
                responses[c_idx] = disc.predict(xv_padded)

            if b == 1:
                first_responses = responses.copy()

            max_res = responses.max()
            if max_res == 0:
                break

            winners = np.flatnonzero(responses == max_res)
            last_tie = winners
            if len(winners) == 1:
                break

            b += max(1, int(b * 0.2))

        pred_class = int(last_tie[0])
        preds[row_idx] = pred_class

        if first_responses is not None:
            total = float(first_responses.sum())
            if total > 0:
                probs[row_idx] = first_responses / total
            else:
                probs[row_idx] = 0.0
                probs[row_idx, pred_class] = 1.0
        else:
            probs[row_idx] = 0.0
            probs[row_idx, pred_class] = 1.0

    return preds, probs


def evaluate_bloom_range_worker(config, start, end):
    train_time = _ensure_state(config)

    start = int(start)
    end = int(end)
    pid = os.getpid()

    t1 = time.perf_counter()
    yp_clean, yp_clean_prob = _predict_dynamic_with_prob(_X_TEST[start:end])
    infer_time_clean = time.perf_counter() - t1

    yp_adv = {}
    yp_adv_prob = {}
    infer_time_adv = {}

    for atk_name, adv_matrix in _ADV.items():
        t2 = time.perf_counter()
        p, pr = _predict_dynamic_with_prob(adv_matrix[start:end])
        yp_adv[atk_name] = p
        yp_adv_prob[atk_name] = pr
        infer_time_adv[atk_name] = time.perf_counter() - t2

    return {
        'pid': pid,
        'start': start,
        'end': end,
        'yp_clean': yp_clean,
        'yp_clean_prob': yp_clean_prob,
        'yp_adv': yp_adv,
        'yp_adv_prob': yp_adv_prob,
        'train_time': float(train_time),
        'infer_time_clean': float(infer_time_clean),
        'infer_time_adv': infer_time_adv,
    }
