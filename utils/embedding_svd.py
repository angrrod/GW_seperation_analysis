"""Shared configuration and cache identity for CPU preparation and GPU training."""
from copy import deepcopy
import os
from pathlib import Path

from dingo.gw.training.embedding_svd_cache import EmbeddingSVDCache
from .paths import get_embedding_svd_cache_dir, get_waveform_dataset_path, get_asd_dataset_path


def resolve_embedding_inputs(settings):
    result = deepcopy(settings)
    result['data']['waveform_dataset_path'] = str(get_waveform_dataset_path())
    for stage in result['training'].values():
        stage['asd_dataset_path'] = str(get_asd_dataset_path())
    return result


def embedding_cache(settings):
    def identity(filename):
        path = Path(filename).resolve()
        stat = path.stat()
        return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    recipe = {
        'version': 1,
        'seed': int(os.environ.get('GW_EMBEDDING_SVD_SEED', '42')),
        'data': deepcopy(settings['data']),
        'svd': deepcopy(settings['model']['embedding_kwargs']['svd']),
        'batch_size': settings['training']['stage_0']['batch_size'],
        'waveform_file': identity(settings['data']['waveform_dataset_path']),
        'asd_file': identity(settings['training']['stage_0']['asd_dataset_path']),
    }
    return EmbeddingSVDCache(get_embedding_svd_cache_dir(), recipe)
