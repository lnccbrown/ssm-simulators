import pytest
import numpy as np
from ssms.support_utils.kde_class import LogKDE, bandwidth_silverman
from ssms.basic_simulators.simulator import simulator


@pytest.fixture
def sample_ddm_data():
    return simulator(
        model="ddm", theta=dict(v=1.0, a=1.5, z=0.5, t=0.3), n_samples=1000
    )


def test_logkde_initialization(sample_ddm_data):
    """Test basic initialization of LogKDE class."""
    kde = LogKDE(simulator_data=sample_ddm_data)
    assert kde.simulator_info == sample_ddm_data["metadata"]
    assert not kde.displace_t
    assert kde.auto_bandwidth is True
    assert kde.bandwidth_type == "silverman"


def test_bandwidth_silverman_with_std():
    """Test bandwidth_silverman function."""
    sample = np.array([1.0, 2.0, 3.0])
    bw = bandwidth_silverman(sample)
    assert isinstance(bw, float)
    assert bw > 0


def test_bandwidth_silverman_without_std():
    """Test bandwidth_silverman function."""
    sample = np.array([1.0, 1.0, 1.0])
    bw = bandwidth_silverman(sample, std_cutoff=0.001)
    assert isinstance(bw, float)
    assert np.allclose(bw, 0.0008, atol=0.001)


@pytest.mark.parametrize(
    "std_proc,expected",
    [
        ("restrict", lambda x: x > 0),  # bandwidth should be positive
        ("kill", lambda x: x == 0),  # bandwidth should be zero
    ],
)
def test_bandwidth_silverman_with_std_proc(std_proc, expected):
    """Test bandwidth_silverman function with different std_proc options."""
    # Use a sample with zero standard deviation to test std_proc behavior
    sample = np.array([1.0, 1.0, 1.0])
    bw = bandwidth_silverman(sample, std_cutoff=0.01, std_proc=std_proc)
    print(f"Bandwidth with std_proc={std_proc}: {bw}")
    assert isinstance(bw, float)
    assert expected(bw)


def test_logkde_compute_bandwidths(sample_ddm_data):
    """Test compute_bandwidths method."""
    kde = LogKDE(simulator_data=sample_ddm_data)
    bandwidths = kde.compute_bandwidths()
    assert isinstance(bandwidths, list)
    assert len(bandwidths) == 2  # For two choices (-1, 1)


def test_kde_eval(sample_ddm_data):
    """Test kde_eval method."""
    kde = LogKDE(simulator_data=sample_ddm_data)
    eval_data = {"rts": np.array([0.6, 0.8]), "choices": np.array([1, -1])}
    result = kde.kde_eval(eval_data, lb=-66.774)
    assert isinstance(result, np.ndarray)
    assert len(result) == 2
    assert np.all(result >= -66.774)


def test_kde_sample(sample_ddm_data):
    """Test kde_sample method."""
    kde = LogKDE(simulator_data=sample_ddm_data)
    samples = kde.kde_sample(n_samples=100)
    assert isinstance(samples, dict)
    assert "rts" in samples
    assert "choices" in samples
    assert len(samples["rts"]) == 100
    assert len(samples["choices"]) == 100


def test_displace_t_validation():
    """Test validation of t parameter when displace_t is True."""
    data = {
        "rts": np.array([0.5, 0.6, 0.7, 0.8]),
        "choices": np.array([1, -1, 1, -1]),
        "metadata": {
            "max_t": 20.0,
            "possible_choices": [-1, 1],
            "t": np.array([0.1, 0.2]),  # Different t values
        },
    }
    with pytest.raises(
        ValueError, match="Multiple t values in simulator data. Can't shift."
    ):
        LogKDE(simulator_data=data, displace_t=True)


def test_invalid_data_kde_eval(sample_ddm_data):
    """Test kde_eval with invalid data."""
    kde = LogKDE(simulator_data=sample_ddm_data)
    with pytest.raises(
        ValueError, match="data dictionary must contain either rts or log_rts as keys!"
    ):
        kde.kde_eval({"invalid_key": np.array([0.6])})


def test_non_positive_rt_does_not_collapse_choice_group():
    """Test that a non-positive RT is dropped without discarding its choice group."""
    rng = np.random.default_rng(7)
    n = 200
    rts = rng.lognormal(mean=-0.3, sigma=0.4, size=n)
    choices = np.where(rng.uniform(size=n) < 0.7, 1.0, -1.0)
    rts[0], choices[0] = -0.5, 1.0  # legitimate for an unbounded t-kernel
    data = {
        "rts": rts.reshape(-1, 1),
        "choices": choices.reshape(-1, 1),
        "metadata": {"max_t": 20.0, "possible_choices": [-1, 1]},
    }

    kde = LogKDE(simulator_data=data)

    assert len(kde.bandwidths) == 2
    for bw in kde.bandwidths:
        assert bw != "no_base_data"
        assert np.isfinite(bw) and bw > 0
    for arr in kde.data["rts"]:
        assert np.all(arr > 0)
    for arr in kde.data["log_rts"]:
        assert np.all(np.isfinite(arr))


def test_clean_data_unchanged_by_filter():
    """Test that an all-positive dataset keeps every sample."""
    rng = np.random.default_rng(11)
    n = 200
    rts = rng.lognormal(mean=-0.3, sigma=0.4, size=n)
    choices = np.where(rng.uniform(size=n) < 0.5, 1.0, -1.0)
    data = {
        "rts": rts.reshape(-1, 1),
        "choices": choices.reshape(-1, 1),
        "metadata": {"max_t": 20.0, "possible_choices": [-1, 1]},
    }

    kde = LogKDE(simulator_data=data)

    assert sum(arr.shape[0] for arr in kde.data["rts"]) == n


def test_single_retained_rt_falls_back_to_std_n_1_bandwidth():
    """Test the n=1 limitation: the std_n_1 default is accepted as a bandwidth."""
    rng = np.random.default_rng(3)
    rts = rng.lognormal(mean=-0.3, sigma=0.4, size=20)
    choices = -np.ones(20)
    rts[0], choices[0] = 0.6, 1.0  # the only retained RT for choice 1
    rts[1], choices[1] = -0.2, 1.0  # dropped by the filter
    data = {
        "rts": rts.reshape(-1, 1),
        "choices": choices.reshape(-1, 1),
        "metadata": {"max_t": 20.0, "possible_choices": [-1, 1]},
    }

    kde = LogKDE(simulator_data=data)

    idx = kde.data["choices"].index(1)
    assert kde.data["rts"][idx].shape[0] == 1
    assert np.isclose(kde.bandwidths[idx], bandwidth_silverman(np.array([np.log(0.6)])))
    assert np.isclose(kde.bandwidths[idx], 10.592238410488122)
