"""Prepare resumable embedding bases on CPU, without a scenario or posterior model."""
import argparse
from pathlib import Path
import random
import sys

# Also support `python scripts/prepare_embedding_svd.py` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bilby
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import yaml

from dingo.gw.training.train_builders import build_dataset, build_svd_for_embedding_network
from utils.embedding_svd import embedding_cache, resolve_embedding_inputs
from utils.paths import get_config_dir


def prepare(settings):
    settings = resolve_embedding_inputs(settings)
    svd = settings['model']['embedding_kwargs']['svd']
    if svd.get('no_init', False):
        raise ValueError('This configuration disables embedding SVD initialization.')
    cache = embedding_cache(settings)
    print(f'Embedding cache: {cache.directory}', flush=True)
    with cache.preparing():
        if cache.load_ready(required=False) is not None:
            print('Compatible embedding cache already complete.', flush=True)
            return cache.directory
        seed = cache.recipe['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        bilby.core.utils.random.seed(seed)
        # Read coefficients on demand instead of loading a six-million-row bank.
        wfd = build_dataset(settings['data'], leave_waveforms_on_disk=True)
        try:
            build_svd_for_embedding_network(
                wfd=wfd, data_settings=settings['data'],
                asd_dataset_path=settings['training']['stage_0']['asd_dataset_path'],
                batch_size=settings['training']['stage_0']['batch_size'],
                out_dir=str(cache.directory), checkpoint=cache, **svd)
            cache.publish(wfd.domain.min_idx)
        finally:
            if wfd.file_handle is not None:
                wfd.file_handle.close()
        print(f'Embedding cache ready for GPU training: {cache.directory}', flush=True)
    return cache.directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['full', 'copula', 'both'], default='both')
    parser.add_argument('--settings-file', type=Path,
                        help='Explicit training YAML; overrides --model.')
    parser.add_argument('--blas-threads', type=int, default=1)
    args = parser.parse_args()
    if args.blas_threads < 1:
        parser.error('--blas-threads must be positive')
    files = ([args.settings_file] if args.settings_file else [
        get_config_dir() / f'training_{model}.yml'
        for model in (['full', 'copula'] if args.model == 'both' else [args.model])])
    torch.set_num_threads(args.blas_threads)
    with threadpool_limits(limits=args.blas_threads):
        for path in files:
            print(f'Preparing embedding for {path}', flush=True)
            with path.open() as stream:
                prepare(yaml.safe_load(stream))


if __name__ == '__main__':
    main()
