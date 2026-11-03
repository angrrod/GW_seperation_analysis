"""Path regression checks; no training, full waveform bank, or GPU required.

Run: python -m unittest discover -s scripts/tests -p test_report_paths.py -v
"""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import h5py

SCRIPT = Path(__file__).resolve().parents[1] / 'report_dingo_pipeline.py'
SPEC = importlib.util.spec_from_file_location('report_dingo_pipeline', SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class TestReportPaths(unittest.TestCase):
    def test_local_default_uses_checkout_even_from_another_cwd(self):
        with patch.dict(os.environ, {}, clear=True):
            paths = report.resolve_report_paths()
        self.assertEqual(Path(paths['config_dir']), report.ROOT / 'configs')
        self.assertEqual(Path(paths['data_dir']), report.ROOT / 'Output/Dingo/data')

    def test_vsc_defaults_separate_data_and_scratch(self):
        with patch.dict(os.environ, {'VSC_DATA': '/data/user', 'VSC_SCRATCH': '/scratch/user'}, clear=True):
            paths = report.resolve_report_paths()
        self.assertEqual(paths['config_dir'], '/data/user/GW_separation/configs')
        self.assertEqual(paths['data_dir'], '/scratch/user/GW_separation/data')

    def test_environment_and_cli_overrides(self):
        env = {'VSC_DATA': '/data/user', 'VSC_SCRATCH': '/scratch/user',
               'GW_INP_DIR': '/data/custom', 'GW_DATA_DIR': '/scratch/custom'}
        with patch.dict(os.environ, env, clear=True):
            paths = report.resolve_report_paths()
            explicit = report.resolve_report_paths('/tmp/configs', '/tmp/datasets')
        self.assertEqual(paths['config_dir'], '/data/custom/configs')
        self.assertEqual(paths['data_dir'], '/scratch/custom')
        self.assertEqual(explicit['config_dir'], '/tmp/configs')
        self.assertEqual(explicit['data_dir'], '/tmp/datasets')

    def test_external_configs_and_hdf5_metadata_work_in_main(self):
        with tempfile.TemporaryDirectory(prefix='gw-report-path-test-') as temp:
            root = Path(temp)
            work = root / 'data/GW_separation'
            configs = work / 'configs'
            datasets = root / 'scratch/GW_separation/data'
            configs.mkdir(parents=True)
            datasets.mkdir(parents=True)
            names = ('training_copula.yml', 'training_full.yml',
                     'waveform_dataset_settings.yml', 'asd_dataset_settings.yml')
            for name in names:
                shutil.copyfile(report.ROOT / 'configs' / name, configs / name)
            for name in ('waveform_dataset.hdf5', 'asd_dataset.hdf5'):
                with h5py.File(datasets / name, 'w') as output:
                    output.create_dataset('metadata_probe', shape=(2, 3), dtype='f4')
            env = {'GW_INP_DIR': str(work), 'GW_DATA_DIR': str(datasets)}
            models = {name: {'sections': {'probe': 1}} for name in ('copula', 'full')}
            with patch.dict(os.environ, env, clear=True), \
                 patch('sys.argv', [str(SCRIPT), '--skip-profile']), \
                 patch.object(report, 'probe_cpu_preparation', return_value={}), \
                 patch.object(report, 'count_models', return_value=models), \
                 patch.object(report, 'write_markdown'), \
                 patch.object(report, 'make_diagram'), \
                 patch('sys.stdout', new_callable=io.StringIO):
                report.main()
            output_dir = work / 'Output/HPC_diagnostics/pipeline_report'
            result = json.loads((output_dir / 'pipeline_report.json').read_text())
            self.assertEqual(result['paths']['config_dir'], str(configs))
            self.assertEqual(result['paths']['data_dir'], str(datasets))
            for name in names:
                self.assertEqual(result['config_paths'][name], str(configs / name))
                self.assertEqual(result['config_sha256']['configs/' + name],
                                 hashlib.sha256((configs / name).read_bytes()).hexdigest())
            self.assertTrue(all(item['exists'] for item in result['local_artifacts']))
            self.assertEqual(result['local_artifacts'][0]['datasets'][0]['shape'], [2, 3])
            self.assertTrue((output_dir / 'tensor_sizes.csv').is_file())

    def test_missing_configs_give_actionable_error_before_probes(self):
        with tempfile.TemporaryDirectory(prefix='gw-report-missing-') as temp, \
             patch('sys.argv', [str(SCRIPT), '--config-dir', temp]), \
             patch.object(report, 'probe_cpu_preparation') as probe, \
             patch('sys.stderr', new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as error:
                report.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn('Missing configuration:', stderr.getvalue())
            self.assertIn('--config-dir', stderr.getvalue())
            probe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
