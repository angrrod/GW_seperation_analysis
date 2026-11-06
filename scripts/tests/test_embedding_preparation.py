"""Path/signature agreement and GPU cache-only initialization contract."""
import ast
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from utils.embedding_svd import embedding_cache, resolve_embedding_inputs
from utils.paths import get_embedding_svd_cache_dir


ROOT = Path(__file__).resolve().parents[2]


class TestEmbeddingPreparation(unittest.TestCase):
    def test_scratch_paths_shared_recipe_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'VSC_SCRATCH': tmp}, clear=True):
            data_dir = Path(tmp) / 'GW_separation/data'
            data_dir.mkdir(parents=True)
            for filename in ('waveform_dataset.hdf5', 'asd_dataset.hdf5'):
                (data_dir / filename).write_bytes(b'input')
            settings = [resolve_embedding_inputs(yaml.safe_load(
                (ROOT / f'configs/training_{model}.yml').read_text()))
                for model in ('full', 'copula')]
            self.assertEqual(get_embedding_svd_cache_dir(), data_dir / 'embedding_svd')
            caches = [embedding_cache(value) for value in settings]
            self.assertEqual(caches[0].directory, caches[1].directory)
            changed = copy.deepcopy(settings[0])
            changed['model']['embedding_kwargs']['svd']['size'] += 1
            self.assertNotEqual(caches[0].signature, embedding_cache(changed).signature)
            (data_dir / 'asd_dataset.hdf5').write_bytes(b'new input')
            self.assertNotEqual(caches[0].signature, embedding_cache(settings[0]).signature)

    def test_explicit_cluster_mounts_override_generic_scratch_alias(self):
        from utils.paths import get_vsc_scratch_dir
        for cluster, expected in (
            ('wice', '/lustre1/scratch/371/vsc37106'),
            ('mindwell', '/gpfs1/scratch/371/vsc37106')):
            with self.subTest(cluster=cluster), patch.dict(os.environ, {
                'SLURM_CLUSTER_NAME': cluster,
                'VSC_SCRATCH': '/scratch/leuven/371/vsc37106'}, clear=True):
                self.assertEqual(get_vsc_scratch_dir(), Path(expected))
                # Test path propagation without attempting to create HPC folders locally.
                with patch.object(Path, 'mkdir'):
                    self.assertEqual(get_embedding_svd_cache_dir(),
                                     Path(expected) / 'GW_separation/data/embedding_svd')
                    with patch.dict(os.environ, {'GW_DATA_DIR': '/tmp/custom-data'}):
                        self.assertEqual(get_embedding_svd_cache_dir(),
                                         Path('/tmp/custom-data/embedding_svd'))

    def test_gpu_wrapper_requires_cache_and_never_builds(self):
        source = ast.parse((ROOT / 'Pipeline/Dingo_NF_pipeline.py').read_text())
        cls = next(node for node in source.body if isinstance(node, ast.ClassDef)
                   and node.name == 'DINGO_pipeline')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_prepare_training_new_with_svd_cache')
        original_builder = Mock(side_effect=AssertionError('must not build'))
        pipeline = SimpleNamespace(build_svd_for_embedding_network=original_builder)
        pipeline.prepare_training_new = Mock(side_effect=lambda **kwargs:
            pipeline.build_svd_for_embedding_network())
        scope = {'patch': patch, 'dingo_train_pipeline': pipeline}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'wrapper', 'exec'), scope)
        pm = SimpleNamespace(train_settings={'model': {'embedding_kwargs': {
            'svd': {'size': 2}}}}, train_dir='run', local_settings={}, logger=Mock())
        cache = SimpleNamespace(directory='scratch/cache', load_ready=Mock(
            side_effect=FileNotFoundError('prepare first')))
        with patch('utils.embedding_svd.embedding_cache', return_value=cache):
            with self.assertRaises(FileNotFoundError):
                scope[method.name](pm)
            pipeline.prepare_training_new.assert_not_called()
            matrices = [object(), object()]
            cache.load_ready = Mock(return_value=matrices)
            self.assertIs(scope[method.name](pm), matrices)
        self.assertIs(pipeline.build_svd_for_embedding_network, original_builder)
        original_builder.assert_not_called()


if __name__ == '__main__':
    unittest.main()
