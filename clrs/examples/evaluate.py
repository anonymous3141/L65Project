"""Evaluate a given model on an algorithmic test set."""

import functools
import os
import shutil
from typing import Any, Dict, List, Optional

from absl import logging
import clrs
import jax
import numpy as np
import requests
import tensorflow as tf
import hydra

FLAGS = None


PRED_AS_INPUT_ALGOS = [
    "binary_search",
    "minimum",
    "find_maximum_subarray",
    "find_maximum_subarray_kadane",
    "matrix_chain_order",
    "lcs_length",
    "optimal_bst",
    "activity_selector",
    "task_scheduling",
    "naive_string_matcher",
    "kmp_matcher",
    "jarvis_march",
]


def unpack(v):
    try:
        return v.item()  # DeviceArray
    except (AttributeError, ValueError):
        return v


def _iterate_sampler(sampler, batch_size):
    while True:
        yield sampler.next(batch_size)


def _maybe_download_dataset(dataset_path):
    """Download CLRS30 dataset if needed."""
    dataset_folder = os.path.join(dataset_path, clrs.get_clrs_folder())
    if os.path.isdir(dataset_folder):
        logging.info("Dataset found at %s. Skipping download.", dataset_folder)
        return dataset_folder
    logging.info("Dataset not found in %s. Downloading...", dataset_folder)

    clrs_url = clrs.get_dataset_gcp_url()
    request = requests.get(clrs_url, allow_redirects=True)
    clrs_file = os.path.join(dataset_path, os.path.basename(clrs_url))
    os.makedirs(dataset_folder)
    open(clrs_file, "wb").write(request.content)
    shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
    os.remove(clrs_file)
    return dataset_folder


def make_sampler(
    length: int,
    rng: Any,
    algorithm: str,
    split: str,
    batch_size: int,
    multiplier: int,
    randomize_pos: bool,
    enforce_pred_as_input: bool,
    enforce_permutations: bool,
    chunked: bool,
    chunk_length: int,
    sampler_kwargs: Dict[str, Any],
):
    """Create a sampler with given options.

    Args:
      length: Size of samples (i.e., number of nodes in the graph).
        A length of -1 will mean that the benchmark
        dataset (for the given split) is used. Positive sizes will instantiate
        samplers of the corresponding size.
      rng: Numpy random state.
      algorithm: The name of the algorithm to sample from.
      split: 'train', 'val' or 'test'.
      batch_size: Samples per batch.
      multiplier: Integer multiplier for the number of samples in the dataset,
        only used for positive sizes. Negative multiplier means infinite samples.
      randomize_pos: Whether to randomize the `pos` input.
      enforce_pred_as_input: Whether to convert fixed pred_h hints to inputs.
      enforce_permutations: Whether to enforce permutation pointers.
      chunked: Whether to chunk the dataset.
      chunk_length: Unroll length of chunks, if `chunked` is True.
      sampler_kwargs: Extra args passed to the sampler.
    Returns:
      A sampler (iterator), the number of samples in the iterator (negative
      if infinite samples), and the spec.
    """
    if length < 0:  # load from file
        dataset_folder = _maybe_download_dataset(FLAGS.dataset_path)
        sampler, num_samples, spec = clrs.create_dataset(
            folder=dataset_folder,
            algorithm=algorithm,
            batch_size=batch_size,
            split=split,
        )
        sampler = sampler.as_numpy_iterator()
    else:
        num_samples = clrs.CLRS30[split]["num_samples"] * multiplier
        sampler, spec = clrs.build_sampler(
            algorithm,
            seed=rng.randint(2**32),
            num_samples=num_samples,
            length=length,
            **sampler_kwargs,
        )
        sampler = _iterate_sampler(sampler, batch_size)

    if randomize_pos:
        sampler = clrs.process_random_pos(sampler, rng)
    if enforce_pred_as_input and algorithm in PRED_AS_INPUT_ALGOS:
        spec, sampler = clrs.process_pred_as_input(spec, sampler)
    spec, sampler = clrs.process_permutations(spec, sampler, enforce_permutations)
    if chunked:
        sampler = clrs.chunkify(sampler, chunk_length)
    return sampler, num_samples, spec


def make_multi_sampler(sizes, rng, **kwargs):
    """Create a sampler with cycling sample sizes."""
    ss = []
    tot_samples = 0
    for length in sizes:
        sampler, num_samples, spec = make_sampler(length, rng, **kwargs)
        ss.append(sampler)
        tot_samples += num_samples

    def cycle_samplers():
        while True:
            for s in ss:
                yield next(s)

    return cycle_samplers(), tot_samples, spec


def _concat(dps, axis):
    return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)


def log_debug_infos(debug_infos):
    for debug_info in debug_infos:
        logging.info("New example")
        logging.info(f"Truth name: {debug_info['truth name']}")
        truth_data_list = debug_info["truth data"].tolist()
        pred_data_list = debug_info["pred data"].tolist()
        n = len(truth_data_list)
        for i in range(n):
            truth = truth_data_list[i]
            pred = pred_data_list[i]
            if truth != pred:
                mismatch_indices = [
                    idx for idx, (t, p) in enumerate(zip(truth, pred)) if t != p
                ]
                logging.info(f"Truth: {truth}")
                logging.info(f"Pred: {pred}")
                logging.info(f"<-- Mismatch at indices {mismatch_indices}")
            else:
                logging.info(f"Truth: {truth} \n Pred: {pred}")


def collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras):
    """Collect batches of output and hint preds and evaluate them."""
    processed_samples = 0
    preds = []
    outputs = []
    while processed_samples < sample_count:
        feedback = next(sampler)
        batch_size = feedback.outputs[0].data.shape[0]
        outputs.append(feedback.outputs)
        new_rng_key, rng_key = jax.random.split(rng_key)
        cur_preds, _, aux_info = predict_fn(new_rng_key, feedback.features)
        preds.append(cur_preds)
        processed_samples += batch_size
    outputs = _concat(outputs, axis=0)
    preds = _concat(preds, axis=0)
    logging.info(f"Outputs length: {len(outputs)}")
    out, debug_infos = clrs.evaluate(outputs, preds)
    log_debug_infos(debug_infos)
    if extras:
        out.update(extras)
    return {k: unpack(v) for k, v in out.items()}, aux_info


def model_weight_norms(params):
    norm_sq_sum = 0
    size_sum = 0
    for tensor in jax.tree_util.tree_leaves(params):
        norm_sq_sum += jax.numpy.linalg.norm(tensor) ** 2
        size_sum += tensor.size
    return jax.numpy.sqrt(norm_sq_sum / size_sum)


def create_samplers(
    rng,
    train_lengths: List[int],
    *,
    algorithms: Optional[List[str]] = None,
    val_lengths: Optional[List[int]] = None,
    test_lengths: Optional[List[int]] = None,
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    test_batch_size: int = 32,
):
    """Create samplers for training, validation and testing.

    Args:
      rng: Numpy random state.
      train_lengths: list of training lengths to use for each algorithm.
      algorithms: list of algorithms to generate samplers for. Set to
          FLAGS.algorithms if not provided.
      val_lengths: list of lengths for validation samplers for each algorithm. Set
          to maxumim training length if not provided.
      test_lengths: list of lengths for test samplers for each algorithm. Set to
          [-1] to use the benchmark dataset if not provided.
      train_batch_size: batch size for training samplers.
      val_batch_size: batch size for validation samplers.
      test_batch_size: batch size for test samplers.

    Returns:
      Tuple of:
        train_samplers: list of samplers for training.
        val_samplers: list of samplers for validation.
        val_sample_counts: list of sample counts for validation.
        test_samplers: list of samplers for testing.
        test_sample_counts: list of sample counts for testing.
        spec_list: list of specs for each algorithm.

    """

    train_samplers = []
    val_samplers = []
    val_sample_counts = []
    test_samplers = []
    test_sample_counts = []
    spec_list = []

    algorithms = algorithms or FLAGS.algorithms
    for algo_idx, algorithm in enumerate(algorithms):
        # Make full dataset pipeline run on CPU (including prefetching).
        with tf.device("/cpu:0"):
            if algorithm in ["naive_string_matcher", "kmp_matcher"]:
                # Fixed haystack + needle; variability will be in needle
                # Still, for chunked training, we maintain as many samplers
                # as train lengths, since, for each length there is a separate state,
                # and we must keep the 1:1 relationship between states and samplers.
                max_length = max(train_lengths)
                if max_length > 0:  # if < 0, we are using the benchmark data
                    max_length = (max_length * 5) // 4
                train_lengths = [max_length]
                if FLAGS.chunked_training:
                    train_lengths = train_lengths * len(train_lengths)

            logging.info("Creating samplers for algo %s", algorithm)

            p = tuple([0.1 + 0.1 * i for i in range(9)])
            if p and algorithm in [
                "articulation_points",
                "bridges",
                "mst_kruskal",
                "bipartite_matching",
            ]:
                # Choose a lower connection probability for the above algorithms,
                # otherwise trajectories are very long
                p = tuple(np.array(p) / 2)
            length_needle = FLAGS.length_needle
            sampler_kwargs = dict(p=p, length_needle=length_needle)
            if length_needle == 0:
                sampler_kwargs.pop("length_needle")

            common_sampler_args = dict(
                algorithm=algorithms[algo_idx],
                rng=rng,
                enforce_pred_as_input=FLAGS.enforce_pred_as_input,
                enforce_permutations=FLAGS.enforce_permutations,
                chunk_length=FLAGS.chunk_length,
            )

            train_args = dict(
                sizes=train_lengths,
                split="train",
                batch_size=train_batch_size,
                multiplier=-1,
                randomize_pos=FLAGS.random_pos,
                chunked=FLAGS.chunked_training,
                sampler_kwargs=sampler_kwargs,
                **common_sampler_args,
            )
            #train_sampler, _, spec = make_multi_sampler(**train_args)

            mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]["num_samples_multiplier"]
            val_args = dict(
                sizes=val_lengths or [np.amax(train_lengths)],
                split="val",
                batch_size=val_batch_size,
                multiplier=2 * mult,
                randomize_pos=FLAGS.random_pos,
                chunked=False,
                sampler_kwargs=sampler_kwargs,
                **common_sampler_args,
            )
            #val_sampler, val_samples, spec = make_multi_sampler(**val_args)

            test_args = dict(
                sizes=test_lengths or [-1],
                split="test",
                batch_size=test_batch_size,
                multiplier=2 * mult,
                randomize_pos=False,
                chunked=False,
                sampler_kwargs={},
                **common_sampler_args,
            )
            test_sampler, test_samples, spec = make_multi_sampler(**test_args)

        spec_list.append(spec)
        #train_samplers.append(train_sampler)
        #val_samplers.append(val_sampler)
        #val_sample_counts.append(val_samples)
        test_samplers.append(test_sampler)
        test_sample_counts.append(test_samples)

    return (
        train_samplers,
        val_samplers,
        val_sample_counts,
        test_samplers,
        test_sample_counts,
        spec_list,
    )


@hydra.main(config_path=".", config_name="config", version_base=None)
def evaluate_saved_model(cfg):
    global FLAGS
    FLAGS = cfg
    print(FLAGS)

    checkpoint_dirs = []
    if "differential_mpnn_maxmax" in FLAGS.processor_type:
        checkpoint_dirs.append(
            "2025-03-17 22:17:26-['insertion_sort']-differential_mpnn_maxmax"
        )
    if "mpnn" in FLAGS.processor_type:
        checkpoint_dirs.append("2025-03-18 09:01:59-['insertion_sort']-mpnn")
    
    if len(checkpoint_dirs) == 0:
        raise ValueError("Processor type not in {differential_mpnn_maxmax, mpnn}.")

    model_paths = [
        os.path.join(os.path.join(FLAGS.checkpoint_path, checkpoint_dir), "best.pkl")
        for checkpoint_dir in checkpoint_dirs
    ]

    for model_path in model_paths:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

    if FLAGS.hint_mode == "encoded_decoded":
        encode_hints = True
        decode_hints = True
    elif FLAGS.hint_mode == "decoded_only":
        encode_hints = False
        decode_hints = True
    elif FLAGS.hint_mode == "none":
        encode_hints = False
        decode_hints = False
    else:
        raise ValueError("Hint mode not in {encoded_decoded, decoded_only, none}.")

    train_lengths = [int(x) for x in FLAGS.train_lengths]

    rng = np.random.RandomState(FLAGS.seed)
    rng_key = jax.random.PRNGKey(rng.randint(2**32))

    # Create samplers
    (
        _,
        _,
        _,
        test_samplers,
        test_sample_counts,
        spec_list,
    ) = create_samplers(
        rng=rng,
        train_lengths=train_lengths,
        algorithms=FLAGS.algorithms,
        val_lengths=[np.amax(train_lengths)],
        test_lengths=[-1],
        train_batch_size=FLAGS.batch_size,
    )

    processor_factory = clrs.get_processor_factory(
        FLAGS.processor_type,
        use_ln=FLAGS.use_ln,
        nb_triplet_fts=FLAGS.nb_triplet_fts,
        nb_heads=FLAGS.nb_heads,
        mpnn_aggregator=FLAGS.mpnn_processor_aggregator,
    )
    model_params = dict(
        processor_factory=processor_factory,
        hidden_dim=FLAGS.hidden_size,
        encode_hints=encode_hints,
        decode_hints=decode_hints,
        encoder_init=FLAGS.encoder_init,
        use_lstm=FLAGS.use_lstm,
        learning_rate=FLAGS.learning_rate,
        grad_clip_max_norm=FLAGS.grad_clip_max_norm,
        checkpoint_path=FLAGS.checkpoint_path,
        freeze_processor=FLAGS.freeze_processor,
        dropout_prob=FLAGS.dropout_prob,
        hint_teacher_forcing=FLAGS.hint_teacher_forcing,
        hint_repred_mode=FLAGS.hint_repred_mode,
        nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
    )

    eval_model = clrs.models.BaselineModel(
        spec=spec_list,
        dummy_trajectory=[next(t) for t in test_samplers],
        **model_params,
    )

    for algo_idx in range(len(test_samplers)):
        common_extras = {"algorithm": FLAGS.algorithms[algo_idx]}

        new_rng_key, rng_key = jax.random.split(rng_key)
        
        for model_path in model_paths:

            eval_model.restore_model(model_path, only_load_processor=False)

            test_stats, final_test_aux_info = collect_and_eval(
                test_samplers[algo_idx],
                functools.partial(eval_model.predict, algorithm_index=algo_idx, return_hints=True),
                test_sample_counts[algo_idx],
                new_rng_key,
                extras=common_extras,
            )
            logging.info(
                "(test model %s) algo %s : %s", model_path, FLAGS.algorithms[algo_idx], test_stats
            )


if __name__ == "__main__":
    evaluate_saved_model()
