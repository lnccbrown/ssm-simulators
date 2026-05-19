"""Tests for TaskEnvironment protocol, built-in bandits, and TaskConfig."""

import numpy as np
import pytest

from ssms.rl.env import GaussianBandit, TaskConfig, TaskEnvironment, TwoArmedBandit


class TestTwoArmedBandit:
    def test_reward_statistics(self):
        """With reward_probabilities=[0.8, 0.2], empirical rates should match."""
        bandit = TwoArmedBandit(reward_probabilities=[0.8, 0.2])
        bandit.reset(rng=np.random.default_rng(42))

        n = 10_000
        rewards_0 = [bandit.generate_reward(0, t) for t in range(n)]
        bandit.reset(rng=np.random.default_rng(43))
        rewards_1 = [bandit.generate_reward(1, t) for t in range(n)]

        assert np.mean(rewards_0) == pytest.approx(0.8, abs=0.02)
        assert np.mean(rewards_1) == pytest.approx(0.2, abs=0.02)

    def test_reproducibility(self):
        """Same seed → same reward sequence."""
        bandit = TwoArmedBandit()
        bandit.reset(rng=np.random.default_rng(99))
        seq1 = [bandit.generate_reward(0, t) for t in range(50)]
        bandit.reset(rng=np.random.default_rng(99))
        seq2 = [bandit.generate_reward(0, t) for t in range(50)]
        assert seq1 == seq2

    def test_different_seeds(self):
        """Different seeds → (very likely) different sequences."""
        bandit = TwoArmedBandit()
        bandit.reset(rng=np.random.default_rng(1))
        seq1 = [bandit.generate_reward(0, t) for t in range(50)]
        bandit.reset(rng=np.random.default_rng(2))
        seq2 = [bandit.generate_reward(0, t) for t in range(50)]
        assert seq1 != seq2

    def test_invalid_reward_probabilities_out_of_range(self):
        with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
            TwoArmedBandit(reward_probabilities=[1.5, 0.3])

    def test_invalid_reward_probabilities_length_mismatch(self):
        with pytest.raises(ValueError, match="must match choices length"):
            TwoArmedBandit(reward_probabilities=[0.7], choices=[0, 1])

    def test_reset_required(self):
        bandit = TwoArmedBandit()
        with pytest.raises(RuntimeError, match="Call reset"):
            bandit.generate_reward(0, 0)

    def test_protocol_compliance(self):
        assert isinstance(TwoArmedBandit(), TaskEnvironment)

    def test_extra_fields_empty(self):
        bandit = TwoArmedBandit()
        assert bandit.extra_fields == []
        bandit.reset()
        assert bandit.get_extra_data(0) == {}

    def test_n_choices_and_choices(self):
        bandit = TwoArmedBandit(choices=[0, 1])
        assert bandit.n_choices == 2
        assert bandit.choices == [0, 1]

    def test_choices_returns_copy(self):
        bandit = TwoArmedBandit(choices=[0, 1])
        c = bandit.choices
        c.append(999)
        assert bandit.choices == [0, 1]


class TestGaussianBandit:
    def test_reward_statistics(self):
        bandit = GaussianBandit(reward_means=[1.5, -0.5], reward_sds=[0.2, 0.8])
        bandit.reset(rng=np.random.default_rng(42))

        n = 20_000
        rewards_0 = np.array([bandit.generate_reward(0, t) for t in range(n)])
        bandit.reset(rng=np.random.default_rng(43))
        rewards_1 = np.array([bandit.generate_reward(1, t) for t in range(n)])

        assert rewards_0.mean() == pytest.approx(1.5, abs=0.01)
        assert rewards_0.std() == pytest.approx(0.2, abs=0.01)
        assert rewards_1.mean() == pytest.approx(-0.5, abs=0.02)
        assert rewards_1.std() == pytest.approx(0.8, abs=0.02)

    def test_reproducibility(self):
        bandit = GaussianBandit()
        bandit.reset(rng=np.random.default_rng(99))
        seq1 = [bandit.generate_reward(0, t) for t in range(50)]
        bandit.reset(rng=np.random.default_rng(99))
        seq2 = [bandit.generate_reward(0, t) for t in range(50)]
        assert seq1 == seq2

    def test_invalid_reward_means_length_mismatch(self):
        with pytest.raises(ValueError, match="reward_means length"):
            GaussianBandit(reward_means=[1.0], choices=[0, 1])

    def test_invalid_reward_sds_length_mismatch(self):
        with pytest.raises(ValueError, match="reward_sds length"):
            GaussianBandit(reward_sds=[1.0], choices=[0, 1])

    def test_invalid_reward_sd_non_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            GaussianBandit(reward_sds=[1.0, 0.0])

    def test_reset_required(self):
        bandit = GaussianBandit()
        with pytest.raises(RuntimeError, match="Call reset"):
            bandit.generate_reward(0, 0)

    def test_protocol_compliance(self):
        assert isinstance(GaussianBandit(), TaskEnvironment)

    def test_n_choices_and_choices(self):
        bandit = GaussianBandit(choices=[-1, 1])
        assert bandit.n_choices == 2
        assert bandit.choices == [-1, 1]

    def test_choices_returns_copy(self):
        bandit = GaussianBandit(choices=[0, 1])
        c = bandit.choices
        c.append(999)
        assert bandit.choices == [0, 1]


class TestTaskConfig:
    def test_builds_bernoulli(self):
        env = TaskConfig(
            reward_type="bernoulli", reward_probabilities=[0.6, 0.4]
        ).build_environment()
        assert isinstance(env, TwoArmedBandit)
        assert env.n_choices == 2

    def test_default_build(self):
        env = TaskConfig().build_environment()
        assert isinstance(env, TwoArmedBandit)
        assert env.n_choices == 2

    def test_builds_gaussian(self):
        env = TaskConfig(
            reward_type="gaussian",
            reward_means=[2.0, -1.0],
            reward_sds=[0.5, 1.5],
        ).build_environment()
        assert isinstance(env, GaussianBandit)
        assert env.n_choices == 2

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported reward_type"):
            TaskConfig(reward_type="exponential").build_environment()

    def test_custom_choices(self):
        env = TaskConfig(
            n_arms=3,
            reward_probabilities=[0.5, 0.3, 0.2],
            choices=[10, 20, 30],
        ).build_environment()
        assert env.choices == [10, 20, 30]
        assert env.n_choices == 3

    def test_gaussian_custom_choices(self):
        env = TaskConfig(
            n_arms=3,
            reward_type="gaussian",
            reward_means=[2.0, 1.0, 0.0],
            reward_sds=[0.2, 0.4, 0.6],
            choices=[10, 20, 30],
        ).build_environment()
        assert env.choices == [10, 20, 30]
        assert env.n_choices == 3
