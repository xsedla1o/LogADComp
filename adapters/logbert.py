import json
import os
import sys
import time
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Optional, Tuple, List, Iterable

import numpy as np
import optuna
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader

from dataloader import NdArr, DataLoader
from sempca.const import device
from sempca.utils import get_logger
from sempca.utils import tqdm
from utils import calculate_metrics, Timed
from .base import ModelPaths, LogADCompAdapter

sys.path.append(str(Path(__file__).parent.parent / "logbert"))

from bert_pytorch.dataset import LogDataset, WordVocab
from bert_pytorch.predict_log import compute_anomaly_bool
from bert_pytorch.dataset.sample import fixed_window_data
from bert_pytorch.model import BERTLog
from bert_pytorch import Predictor, Trainer

CACHED_DATASET_KEYS = [
    "train_path",
    "valid_path",
    "test_normal_path",
    "test_anomaly_path",
]
CACHED_PATH_KEYS = CACHED_DATASET_KEYS + ["vocab_path"]


class LogBERTAdapter(LogADCompAdapter):
    def __init__(self):
        super().__init__()
        self.log = get_logger("LogBERTAdapter")
        self.vocab_size = None
        self.threshold = None
        self.params = None

    def set_paths(self, paths: ModelPaths):
        """Set the paths for the LogBERT adapter."""
        output_d = paths.cache
        model_d = output_d / "bert"
        paths.register_subdir(model_d)
        self.threshold_trials_path = paths.artefacts / "training_trial_results.csv"

        self.o = {
            "device": device,
            "output_dir": output_d,
            "model_dir": model_d,
            "artefact_dir": paths.artefacts,
            "model_path": model_d / "best_bert.pth",
            "model_threshold_path": model_d / "threshold.json",
            "train_vocab": output_d / "train",
            "vocab_path": output_d / "vocab.pkl",
            "scale_path": model_d / "scale.pkl",
            "train_path": output_d / "train",
            "valid_path": output_d / "valid",
            "test_normal_path": output_d / "test_normal",
            "test_anomaly_path": output_d / "test_anomaly",
        }

        # Ensure directories exist
        # Convert paths to strings to keep LogBERT happy
        for k, path in self.o.items():
            if isinstance(path, Path):
                if path.is_dir():
                    self.o[k] = str(path) + "/"
                else:
                    self.o[k] = str(path)

        # For the skip_when_present decorator
        self.config = {k: self.o[k] for k in CACHED_PATH_KEYS}

    @staticmethod
    def transform_representation(loader: DataLoader) -> Tuple[NdArr, NdArr]:
        """Adapts the data from the loader."""
        return loader.get_t_seq_representation()

    def preprocess_split(
        self, x_train: NdArr, x_val: NdArr, x_test: NdArr
    ) -> Tuple[NdArr, NdArr, NdArr]:
        """Preprocesses and writes the instances in the LogBERT format."""
        self.log.info("Preprocessing the instances")
        self._prepare_vocab(x_train, x_val)
        return x_train, x_val, x_test

    def _prepare_vocab(self, x_train, x_val):
        vocab = WordVocab(np.concatenate([x_train, x_val], axis=0))
        self.log.debug("Vocab size: %d", len(vocab))
        self.log.debug("Saving to: %s", self.o["vocab_path"])
        self.vocab_size = len(vocab)
        vocab.save_vocab(self.o["vocab_path"])

    @staticmethod
    def _write_to_file(seqs, out_f):
        """Write a list of instances to a file in LogBERT format."""
        with open(out_f, "w") as out_f:
            for seq in seqs:
                out_f.write(" ".join(map(str, seq)))
                out_f.write("\n")

    def set_params(self, threshold: float = 0.2, num_candidates: int = 6):
        """No hyperparameters to set directly for LogBERT, as they are managed internally."""
        self.o["window_size"] = 128
        self.o["adaptive_window"] = True
        self.o["seq_len"] = 512
        self.o["max_len"] = 512  # for position embedding
        self.o["min_len"] = 0
        self.o["mask_ratio"] = 0.65
        # sample ratio
        self.o["train_ratio"] = 1
        self.o["valid_ratio"] = 0.1
        self.o["test_ratio"] = 1

        # features
        self.o["is_logkey"] = True
        self.o["is_time"] = False

        self.o["hypersphere_loss"] = True
        self.o["hypersphere_loss_test"] = False

        self.o["scale"] = None  # MinMaxScaler()

        # model
        self.o["hidden"] = 256  # embedding size
        self.o["layers"] = 4
        self.o["attn_heads"] = 4

        self.o["epochs"] = 200
        self.o["warm_up_epochs"] = 10
        self.o["n_epochs_stop"] = 10
        self.o["batch_size"] = 32

        self.o["corpus_lines"] = None
        self.o["on_memory"] = True
        self.o["num_workers"] = min(5, int(os.getenv("PBS_NCPUS", os.cpu_count())))
        self.o["lr"] = 1e-3
        self.o["adam_beta1"] = 0.9
        self.o["adam_beta2"] = 0.999
        self.o["adam_weight_decay"] = 0.00
        self.o["with_cuda"] = True
        self.o["cuda_devices"] = None
        self.o["log_freq"] = None

        # predict
        self.o["gaussian_mean"] = 0
        self.o["gaussian_std"] = 1

        # predict tuned
        if self.params is not None:
            self.log.info("Overriding passed values to %s", self.params)
            self.o["num_candidates"] = self.params["num_candidates"]
            self.threshold = self.params["threshold"]
        else:
            self.o["num_candidates"] = num_candidates
            self.threshold = threshold

    def get_trial_objective(
        self, x_train, y_train, x_val, y_val, prev_params: dict = None
    ):
        def objective(trial: optuna.Trial):
            return 0.0

        return objective

    def find_thresholds(self, x_norm, y_true, n_trials=20, max_threshold=0.3):
        """Tune hyperparameters of anomaly detection logic.
        - num_candidates: number of candidates to consider for anomaly detection
        - threshold: threshold of missed predictions to trigger anomaly
        """
        p, model = self._load_predictor_model()
        vocab = WordVocab.load_vocab(p.vocab_path)
        dl, *reassembly_context = self._prepare_sequences(x_norm, p, vocab)

        if self.vocab_size < 256:
            results_df = self._find_thresholds_small(
                y_true,
                p,
                model,
                dl,
                reassembly_context,
                n_trials=n_trials,
                max_threshold=max_threshold,
            )
        else:
            self.log.debug("Finding thresholds for large vocab size")
            results_df = self.find_thresholds_large(
                y_true,
                p,
                model,
                dl,
                reassembly_context,
                n_trials=n_trials,
                max_threshold=max_threshold,
            )
        results_df.to_csv(self.threshold_trials_path, index=False)
        self.log.info("Results saved to %s", self.threshold_trials_path)
        best_params = dict(
            results_df.loc[results_df.f1.idxmax(), ["num_candidates", "threshold"]]
        )
        self.log.info("Best params: %s", best_params)
        return best_params

    def _find_thresholds_small(
        self, y_true, p, model, dl, reassembly_context, n_trials=20, max_threshold=0.3
    ):
        with torch.no_grad():
            inputs, outputs = self._get_raw_outputs(p, model, dl)

        trial_results = []

        def objective(trial: optuna.Trial):
            n_candidates = trial.suggest_int("num_candidates", 1, self.vocab_size)
            pp_results = self._post_process_batches(p, inputs, outputs, n_candidates)

            max_F1 = 0
            for thresh in range(1, int(max_threshold * 100)):
                threshold = thresh / 100  # Threshold ranges from 0.01 to 0.49

                sample_pred_y = compute_anomaly_bool(
                    pp_results, p.get_params(), threshold
                )

                seq_pred_y = self._reassemble_predictions(
                    sample_pred_y, *reassembly_context
                )

                metrics = calculate_metrics(y_true, seq_pred_y)

                trial_results.append(
                    {"num_candidates": n_candidates, "threshold": threshold} | metrics
                )
                max_F1 = max(max_F1, metrics["f1"])
            return max_F1

        study = optuna.create_study(
            study_name="LogBERT_optimization", direction="maximize"
        )
        study.optimize(objective, n_trials=n_trials)
        results_df = pd.DataFrame(trial_results)
        return results_df

    @staticmethod
    def compute_anomaly_bool_threshold_range(results, params, max_threshold):
        is_logkey = params["is_logkey"]
        is_time = params["is_time"]
        detection_results = [[] for _ in range(1, int(max_threshold * 100))]
        for seq_res_gen in results:
            for seq_res in seq_res_gen:
                for thresh in range(0, int(max_threshold * 100) - 1):
                    threshold = thresh / 100
                    detection_results[thresh].append(
                        (
                            is_logkey
                            and seq_res["undetected_tokens"]
                            > seq_res["masked_tokens"] * threshold
                        )
                        or (
                            is_time
                            and seq_res["num_error"]
                            > seq_res["masked_tokens"] * threshold
                        )
                        or (
                            params["hypersphere_loss_test"]
                            and seq_res["deepSVDD_label"]
                        )
                    )
        return [np.asarray(res, dtype=bool) for res in detection_results]

    def find_thresholds_large(
        self, y_true, p, model, dl, reassembly_context, n_trials=20, max_threshold=0.3
    ):
        trial_results = []

        def objective(trial: optuna.Trial):
            n_candidates = trial.suggest_int("num_candidates", 1, self.vocab_size)
            tmp_candidates = p.num_candidates
            p.num_candidates = n_candidates

            with torch.no_grad():
                results = self._predict_helper_iter(p, model, dl)
                sample_pred_ys = self.compute_anomaly_bool_threshold_range(
                    results, p.get_params(), max_threshold
                )

            max_F1 = 0
            for sample_pred_y, int_thr in zip(
                sample_pred_ys, range(1, int(max_threshold * 100))
            ):
                threshold = int_thr / 100
                seq_pred_y = self._reassemble_predictions(
                    sample_pred_y, *reassembly_context
                )

                metrics = calculate_metrics(y_true, seq_pred_y)
                trial_results.append(
                    {"num_candidates": n_candidates, "threshold": threshold} | metrics
                )
                max_F1 = max(max_F1, metrics["f1"])

            p.num_candidates = tmp_candidates
            return max_F1

        study = optuna.create_study(
            study_name="LogBERT_optimization", direction="maximize"
        )
        study.optimize(objective, n_trials=n_trials)
        results_df = pd.DataFrame(trial_results)
        return results_df

    @staticmethod
    def _get_raw_outputs(p: Predictor, model: BERTLog, data_loader: TorchDataLoader):
        """Only get the model outputs without any post-processing.

        Meant for usage in optimize_hyperparameters to avoid re-computing the model outputs.
        """
        total_results = []
        total_inputs = []

        for idx, data in enumerate(tqdm(data_loader, desc="Predicting")):
            data = {key: value.to(p.device) for key, value in data.items()}
            result = model(data["bert_input"], data["time_input"])
            # Cloning tensors is necessary to allow the dataloader threads to die,
            # otherwise program will crash due to too many files being opened.
            # Could be solved instead with `pytorch.multiprocessing
            # .set_sharing_strategy("file_system")`, but this seems simpler.
            total_results.append(
                {
                    k: None if t is None else t.detach().clone()
                    for k, t in result.items()
                }
            )
            total_inputs.append(
                {k: data[k].detach().clone() for k in ["bert_input", "bert_label"]}
            )

        return total_inputs, total_results

    def _post_process_batches(self, p, inputs, results, n_candidates):
        assert len(inputs) == len(results), "Inputs and results must match in length"
        tmp_candidates = p.num_candidates
        p.num_candidates = n_candidates
        total_results = []

        for idx, (data, result) in enumerate(
            tqdm(zip(inputs, results), desc="Post-processing", total=len(inputs))
        ):
            # cls_output: batch_size x hidden_size
            results = self._post_model_process_batch(
                p, data, result["logkey_output"], result["cls_output"]
            )
            total_results.extend(results)

        p.num_candidates = tmp_candidates
        # for hypersphere distance
        return total_results

    def fit(self, x_train, y_train, x_val, y_val):
        """Trains the LogBERT model or loads an existing one if available."""
        t = Trainer(self.o)

        x_train_norm = x_train[y_train == 0]
        x_val_norm = x_val[y_val == 0]

        train_seq_x, train_tim_x, *_ = self.fixed_windows_from_sequences(
            x_train_norm, t.window_size, t.adaptive_window, t.seq_len, t.min_len
        )
        val_seq_x, val_tim_x, *_ = self.fixed_windows_from_sequences(
            x_val_norm, t.window_size, t.adaptive_window, t.seq_len, t.min_len
        )

        t.train_on(train_seq_x, train_tim_x, val_seq_x, val_tim_x)

        xs = np.concatenate([x_train, x_val], axis=0)
        ys = np.concatenate([y_train, y_val], axis=0)

        self.params = self.find_thresholds(xs, ys)
        with open(self.o["model_threshold_path"], "w") as fp:
            json.dump(self.params, fp)
        self.o["num_candidates"] = int(self.params["num_candidates"])
        self.threshold = self.params["threshold"]

    def _get_thresh_params(self) -> Optional[dict]:
        """Get the parameters for thresholding."""
        if not (
            os.path.exists(self.o["model_threshold_path"])
            and os.path.isfile(self.o["model_threshold_path"])
        ):
            return None
        with open(self.o["model_threshold_path"], "r") as fp:
            return json.load(fp)

    def _load_predictor_model(self, options: dict = None) -> Tuple[Predictor, BERTLog]:
        """Load the LogBERT model."""
        if options is None:
            self.params = self._get_thresh_params()
            if self.params is not None:
                self.o["num_candidates"] = int(self.params["num_candidates"])
                self.threshold = self.params["threshold"]
            options = self.o
        p = Predictor(options)
        model: BERTLog = torch.load(p.model_path, weights_only=False)
        model.to(p.device)
        model.eval()

        if p.hypersphere_loss:
            center_dict = torch.load(p.model_dir + "best_center.pt", weights_only=False)
            p.center = center_dict["center"]
            p.radius = center_dict["radius"]
        return p, model

    @staticmethod
    def _reassemble_predictions(
        sample_y_pred: np.ndarray,
        inverse_indices: np.ndarray,
        seq_split_cnts: List[int],
    ) -> np.ndarray:
        ordered_y_pred = sample_y_pred[inverse_indices]

        y_pred = []
        start_idx = 0
        for count in seq_split_cnts:
            sample_pred = ordered_y_pred[start_idx : start_idx + count]
            seq_pred = 1 if sample_pred.any() else 0
            y_pred.append(seq_pred)
            start_idx += count
        return np.array(y_pred, dtype=int)

    def predict(self, x_test):
        """Makes predictions using LogBERT."""
        with torch.no_grad():
            p, model = self._load_predictor_model()
            vocab = WordVocab.load_vocab(p.vocab_path)
            test_dl, *reassembly_context = self._prepare_sequences(x_test, p, vocab)

            with Timed("Running model to get anomaly scores"):
                test_results = self._predict_helper_iter(p, model, test_dl)

            with Timed("Detecting anomalies based on threshold"):
                sample_y_pred = compute_anomaly_bool(
                    chain.from_iterable(test_results), p.get_params(), self.threshold
                )

            return self._reassemble_predictions(sample_y_pred, *reassembly_context)

    @staticmethod
    def fixed_windows_from_sequences(
        sequences, window_size, adaptive_window, seq_len, min_len
    ) -> Tuple[NdArr, NdArr, NdArr, List[int]]:
        """
        Generate log_seqs and tim_seqs directly from a list of instances without
        converting the sequences to a string.

        Each instance is expected to have a 'sequence' attribute that is a list of tokens.
        Tokens should be either a single value (log key) or a two-element structure [log_key, timestamp].

        :param sequences: List of sequences.
        :param window_size: Window size for segmentation.
        :param adaptive_window: Boolean flag for adaptive windowing.
        :param seq_len: Maximum number of tokens per session.
        :param scale: Optional scaler for time sequences (if needed).
        :param min_len: Minimum sequence length required.
        :return: Tuple (log_seqs, tim_seqs) sorted by descending sequence length.
        """
        log_seqs = []
        tim_seqs = []
        seq_split_cnts = []
        skipped = 0

        for seq in sequences:
            log_seq, tim_seq, split_cnt = fixed_window_data(
                seq,
                window_size,
                adaptive_window=adaptive_window,
                seq_len=seq_len,
                min_len=min_len,
            )
            if split_cnt == 0:
                skipped += 1
                continue

            log_seqs += log_seq
            tim_seqs += tim_seq
            seq_split_cnts.append(split_cnt)

        # Convert to numpy arrays (using dtype=object to accommodate variable lengths).
        log_seqs = np.array(log_seqs, dtype=object)
        tim_seqs = np.array(tim_seqs, dtype=object)

        # Sort sequences by their length in descending order.
        lengths = np.array(list(map(len, log_seqs)))
        sorted_indices = np.argsort(-lengths)
        log_seqs = log_seqs[sorted_indices]
        tim_seqs = tim_seqs[sorted_indices]

        inverse_indices = np.empty_like(sorted_indices)
        inverse_indices[sorted_indices] = np.arange(sorted_indices.size)

        print(f"Processed {len(log_seqs)} sequences, skipped {skipped}")
        assert len(log_seqs) == len(tim_seqs)
        assert len(log_seqs) == len(seq_split_cnts)
        return log_seqs, tim_seqs, inverse_indices, seq_split_cnts

    @staticmethod
    def _logkey_detection(p: Predictor, mask_lm_output: Tensor, bert_labels, masks):
        # shape (num_masked, 2) where each row is [batch_index, token_index]
        m_idxs = torch.nonzero(masks, as_tuple=False)
        # extracts all masked token outputs, flattening to (num_masked, V)
        flat_masked_output = mask_lm_output[m_idxs[:, 0], m_idxs[:, 1]]
        flat_masked_labels = bert_labels[m_idxs[:, 0], m_idxs[:, 1]]

        # Compute how many masked tokens each batch element has
        B = mask_lm_output.size(0)
        batch_idx = m_idxs[:, 0]
        counts = torch.bincount(batch_idx, minlength=B)  # shape: (B,)
        max_count = counts.max().item()

        # Compute the position (within each batch) for each masked token.
        offsets = torch.zeros_like(counts)
        if B > 0:
            offsets[1:] = torch.cumsum(counts, dim=0)[:-1]
        # Each token's position is its overall index minus the offset for its batch.
        flat_0_positions = torch.arange(m_idxs.size(0), device=batch_idx.device)
        positions = flat_0_positions - offsets[batch_idx]

        # Allocate padded tensors, shape (B, max_count, V), (B, max_count)
        padded_masked_output = torch.zeros(
            (B, max_count, mask_lm_output.size(-1)), device=mask_lm_output.device
        )
        padded_masked_labels = torch.full(
            (B, max_count), -1, device=flat_masked_labels.device
        )

        # Use advanced indexing to scatter each token into the proper position
        padded_masked_output[batch_idx, positions] = flat_masked_output
        padded_masked_labels[batch_idx, positions] = flat_masked_labels

        # Create valid mask to disregard padding entries
        valid_mask = padded_masked_labels != -1

        undetected_tokens, _ = p.detect_logkey_anomaly_vectorized(
            padded_masked_output, padded_masked_labels, valid_mask
        )
        return undetected_tokens

    def _prepare_sequences(
        self, sequences, p: Predictor, vocab: WordVocab
    ) -> Tuple[TorchDataLoader, np.ndarray, List[int]]:
        """Prepare sequences for prediction."""
        # Convert sequences to the format expected by the model.
        logkey_test, time_test, *reassembly_ctx = self.fixed_windows_from_sequences(
            sequences, p.window_size, p.adaptive_window, p.seq_len, p.min_len
        )

        seq_dataset = LogDataset(
            logkey_test,
            time_test,
            vocab,
            seq_len=p.seq_len,
            corpus_lines=p.corpus_lines,
            on_memory=p.on_memory,
            predict_mode=True,
            mask_ratio=p.mask_ratio,
        )

        data_loader = TorchDataLoader(
            seq_dataset,
            batch_size=p.batch_size,
            num_workers=p.num_workers,
            collate_fn=seq_dataset.collate_fn,
        )
        return data_loader, reassembly_ctx[0], reassembly_ctx[1]

    def _predict_helper(
        self, p: Predictor, model: BERTLog, data_loader: TorchDataLoader
    ):
        total_results = []
        output_cls = []

        telem = defaultdict(float)
        t_begin = time.time()
        for idx, data in enumerate(tqdm(data_loader, desc="Predicting")):
            data = {key: value.to(p.device) for key, value in data.items()}
            ts = time.time()
            result = model(data["bert_input"], data["time_input"])
            telem["forward"] += time.time() - ts

            # cls_output: batch_size x hidden_size
            output_cls += result["cls_output"].tolist()

            tsv = time.time()
            results = self._post_model_process_batch(
                p, data, result["logkey_output"], result["cls_output"]
            )
            total_results.extend(results)

            telem["vectorized detection"] += time.time() - tsv

        t_total = time.time() - t_begin
        telem["unaccounted"] = t_total - sum(telem.values())
        prof_line = ", ".join(
            f"{i + 1}. {k}: {v:.3f}s" for i, (k, v) in enumerate(telem.items())
        )
        print("Profiled: " + prof_line)

        # for hypersphere distance
        return total_results, output_cls

    def _predict_helper_iter(
        self, p: Predictor, model: BERTLog, data_loader: TorchDataLoader
    ):
        for idx, data in enumerate(tqdm(data_loader, desc="Predicting")):
            data = {key: value.to(p.device) for key, value in data.items()}
            result = model(data["bert_input"], data["time_input"])

            results = self._post_model_process_batch(
                p, data, result["logkey_output"], result["cls_output"]
            )

            yield results

    def _post_model_process_batch(
        self,
        p: Predictor,
        data: dict,
        mask_lm_output: torch.Tensor,
        cls_output: torch.Tensor,
    ) -> Iterable[dict]:
        """
        Args:
            p: Predictor
            data: dict containing the input data
            mask_lm_output: batch_size x session_size x vocab_size
            cls_output: batch_size x hidden_size
        """
        # bert_label, time_label: batch_size x session_size
        # in session, some logkeys are masked
        bert_labels = data["bert_label"]

        undetected_tokens = []
        svdd_labels = []
        total_logkeys = torch.sum(data["bert_input"] > 0, dim=1).tolist()

        masks = bert_labels > 0
        masked_tokens = torch.sum(masks, dim=1).tolist()

        if p.is_logkey:
            undetected_tokens = self._logkey_detection(
                p, mask_lm_output, bert_labels, masks
            )

        if p.hypersphere_loss_test:
            # detect by deepSVDD distance
            assert cls_output[0].size() == p.center.size()
            dist = torch.sqrt(torch.sum((cls_output - p.center) ** 2, dim=1))
            svdd_labels = (dist > p.radius).long().tolist()

        return (
            {
                "num_error": 0,
                "undetected_tokens": undetected_tokens[j].item() if p.is_logkey else 0,
                "masked_tokens": masked_tokens[j],
                "total_logkey": total_logkeys[j],
                "deepSVDD_label": svdd_labels[j] if p.hypersphere_loss_test else 0,
            }
            for j in range(len(bert_labels))
        )
