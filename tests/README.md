# Tests

This directory contains tests for the torchtitan project, including unit tests and integration tests.

## Test Structure

- `unit_tests/`: Contains unit tests for individual components
- `integration_tests/`: Contains integration tests that test multiple components together
  - `models.py`: One short multi-GPU training run per module (`olmo2`, `stem_dag`, `engram_olmo2`, `olmo2_moe`) on its `*_debugmodel` config
- `assets/`: Contains test assets and fixtures used by the tests
  - `tokenizer/`: Tokenizer configuration and vocabulary files for testing
  - `c4_test/`: Small C4 sample the debugmodel configs train on

## Running Tests

### Prerequisites

Ensure you have all development dependencies installed:

```bash
pip install -r .ci/docker/requirements-dev.txt
pip install -r requirements.txt
```

### Running Integration Tests

To run the integration tests:

```bash
python -m tests.integration_tests.run_tests <output_dir> [--module MODULE] [--config CONFIG] [--test_suite TEST_SUITE] [--test_name TEST_NAME] [--ngpu NGPU]
```

Arguments:
- `output_dir`: (Required) Directory where test outputs will be stored
- `--module`: (Optional) Model module to use for training (default: "olmo2"). Passed as `MODULE` env var to `run_train.sh`; each test in `models.py` overrides it with its own `--module`.
- `--config`: (Optional) Config function to use for training (default: "olmo2_debugmodel"). Passed as `CONFIG` env var to `run_train.sh`; each test overrides it with its own `--config`.
- `--test_suite`: (Optional) Test suite to run (default and only choice: "models")
- `--test_name`: (Optional) Specific test to run by name (default: "all")
- `--ngpu`: (Optional) Number of GPUs to use for testing (default: 8)

Examples:
```bash
# Run every model test
python -m tests.integration_tests.run_tests test_output

# Run only the STAPLE test
python -m tests.integration_tests.run_tests test_output --test_name stem_dag_fsdp+ep+compile
```

### Running Unit Tests

To run only the unit tests:

```bash
pytest -s tests/unit_tests/
```

### Running Specific Unit Test Files

To run a specific test file:

```bash
pytest -s tests/unit_tests/test_config_manager.py
```

### Running Specific Test Functions in Unit Tests

To run a specific test function:

```bash
pytest -s tests/unit_tests/test_config_manager.py::TestConfigManager::test_cli_overrides
```
