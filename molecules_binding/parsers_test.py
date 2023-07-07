""" test parsers functions on an element of the dataset"""
from molecules_binding.parsers import get_affinities
from molecules_binding.datasets_interaction import structural_info
import pytest
import numpy as np

path_example_index = "example_dataset/index_data_example.2020"
a, b, c = np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0])


@pytest.mark.parametrize("affinity_dict, expected_dict",
                         [(get_affinities(path_example_index), {
                             "3zzf": ["Ki", 0.4, 400.0, "mM", True],
                             "1hvl": ["Ki", 9.95, 112.0, "pM", True],
                             "1zsb": ["Kd", 0.6, 250.0, "mM", True],
                             "4ux4": ["IC50", 7.01, 97.0, "nM", True],
                             "5a3s": ["Kd", 8.68, 2.1, "nM", True],
                             "2uyw": ["Kd", 13.0, 0.1, "pM", True]
                         })])
def test_get_affinities(affinity_dict, expected_dict):
    assert affinity_dict == expected_dict


@pytest.mark.benchmark(group="structural_info")
def test_structural_info(benchmark, espected_structural_info=(90.0, 0.5, 1.0)):
    result = benchmark(structural_info, a, b, c)
    assert result == espected_structural_info
