"""Tests for TaskEnvironment protocol, Bandit, and TaskConfig."""

import numpy as np
import pytest

from ssms.rl.env import Bandit, TaskConfig, TaskEnvironment, registered_tasks


class TestBernoulliBandit:
    def test_reward_statistics_two_arms(self):
        bandit = Bandit.bernoulli(probabilities=[0.8, 0.2])
        bandit.reset(rng=np.random.default_rng(42))

        n = 10_000
        rewards_0 = [bandit.sample_reward(0, t) for t in range(n)]
        bandit.reset(rng=np.random.default_rng(43))
        rewards_1 = [bandit.sample_reward(1, t) for t in range(n)]

        assert np.mean(rewards_0) == pytest.approx(0.8, abs=0.02)
        assert np.mean(rewards_1) == pytest.approx(0.2, abs=0.02)

    def test_reward_statistics_three_arms(self):
        bandit = Bandit.bernoulli(probabilities=[0.8, 0.5, 0.2])
        n = 10_000

        means = []
        for action in range(3):
            bandit.reset(rng=np.random.default_rng(42 + action))
            rewards = [bandit.sample_reward(action, t) for t in range(n)]
            means.append(np.mean(rewards))

        assert means[0] == pytest.approx(0.8, abs=0.02)
        assert means[1] == pytest.approx(0.5, abs=0.02)
        assert means[2] == pytest.approx(0.2, abs=0.02)

    def test_reproducibility(self):
        bandit = Bandit.bernoulli()
        bandit.reset(rng=np.random.default_rng(99))
        seq1 = [bandit.sample_reward(0, t) for t in range(50)]
        bandit.reset(rng=np.random.default_rng(99))
        seq2 = [bandit.sample_reward(0, t) for t in range(50)]
        assert seq1 == seq2

    def test_different_seeds(self):
        bandit = Bandit.bernoulli()
        bandit.reset(rng=np.random.default_rng(1))
        seq1 = [bandit.sample_reward(0, t) for t in range(50)]
        bandit.reset(rng=np.random.default_rng(2))
        seq2 = [bandit.sample_reward(0, t) for t in range(50)]
        assert seq1 != seq2

    def test_invalid_probability_out_of_range(self):
        with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
            Bandit.bernoulli(probabilities=[1.5, 0.3])

    def test_invalid_probability_too_few_arms(self):
        with pytest.raises(ValueError, match="at least 2 arms"):
            Bandit.bernoulli(probabilities=[0.7])

    def test_empty_probabilities_are_invalid(self):
        with pytest.raises(ValueError, match="at least 2 arms"):
            Bandit.bernoulli(probabilities=[])

    def test_response_labels_length_mismatch(self):
        with pytest.raises(ValueError, match="response_labels length"):
            Bandit.bernoulli(probabilities=[0.7, 0.3], response_labels=[0])

    def test_duplicate_response_labels(self):
        with pytest.raises(ValueError, match="response_labels must be unique"):
            Bandit.bernoulli(probabilities=[0.7, 0.3], response_labels=[1, 1])

    def test_reset_required(self):
        bandit = Bandit.bernoulli()
        with pytest.raises(RuntimeError, match="Call reset"):
            bandit.sample_reward(0, 0)

    def test_action_out_of_range(self):
        bandit = Bandit.bernoulli()
        bandit.reset()
        with pytest.raises(ValueError, match="out of range"):
            bandit.sample_reward(2, 0)

    def test_protocol_compliance(self):
        assert isinstance(Bandit.bernoulli(), TaskEnvironment)

    def test_extra_fields_empty(self):
        bandit = Bandit.bernoulli()
        assert bandit.extra_fields == []
        bandit.reset()
        assert bandit.get_extra_data(0) == {}

    def test_n_arms_and_response_labels(self):
        bandit = Bandit.bernoulli(
            probabilities=[0.4, 0.3, 0.2], response_labels=[-1, 0, 1]
        )
        assert bandit.n_arms == 3
        assert bandit.response_labels == [-1, 0, 1]

    def test_response_labels_returns_copy(self):
        bandit = Bandit.bernoulli(response_labels=[0, 1])
        labels = bandit.response_labels
        labels.append(999)
        assert bandit.response_labels == [0, 1]


class TestGaussianRewards:
    def test_reward_statistics_two_arms(self):
        bandit = Bandit.gaussian(means=[1.5, -0.5], sds=[0.2, 0.8])
        bandit.reset(rng=np.random.default_rng(42))

        n = 20_000
        rewards_0 = np.array([bandit.sample_reward(0, t) for t in range(n)])
        bandit.reset(rng=np.random.default_rng(43))
        rewards_1 = np.array([bandit.sample_reward(1, t) for t in range(n)])

        assert rewards_0.mean() == pytest.approx(1.5, abs=0.01)
        assert rewards_0.std() == pytest.approx(0.2, abs=0.01)
        assert rewards_1.mean() == pytest.approx(-0.5, abs=0.02)
        assert rewards_1.std() == pytest.approx(0.8, abs=0.02)

    def test_reward_statistics_three_arms(self):
        bandit = Bandit.gaussian(means=[1.5, 0.5, -0.5], sds=[0.2, 0.4, 0.8])
        n = 20_000

        observed = []
        for action in range(3):
            bandit.reset(rng=np.random.default_rng(42 + action))
            rewards = np.array([bandit.sample_reward(action, t) for t in range(n)])
            observed.append((rewards.mean(), rewards.std()))

        assert observed[0][0] == pytest.approx(1.5, abs=0.01)
        assert observed[0][1] == pytest.approx(0.2, abs=0.01)
        assert observed[1][0] == pytest.approx(0.5, abs=0.01)
        assert observed[1][1] == pytest.approx(0.4, abs=0.01)
        assert observed[2][0] == pytest.approx(-0.5, abs=0.02)
        assert observed[2][1] == pytest.approx(0.8, abs=0.02)

    def test_reproducibility(self):
        bandit = Bandit.gaussian()
        bandit.reset(rng=np.random.default_rng(99))
        seq1 = [bandit.sample_reward(0, t) for t in range(50)]
        bandit.reset(rng=np.random.default_rng(99))
        seq2 = [bandit.sample_reward(0, t) for t in range(50)]
        assert seq1 == seq2

    def test_invalid_means_too_few_arms(self):
        with pytest.raises(ValueError, match="at least 2 arms"):
            Bandit.gaussian(means=[1.0], sds=[1.0])

    def test_empty_means_are_invalid(self):
        with pytest.raises(ValueError, match="at least 2 arms"):
            Bandit.gaussian(means=[], sds=[])

    def test_invalid_sds_length_mismatch(self):
        with pytest.raises(ValueError, match="means length"):
            Bandit.gaussian(means=[1.0, 0.0], sds=[1.0])

    def test_empty_sds_are_invalid(self):
        with pytest.raises(ValueError, match="means length"):
            Bandit.gaussian(means=[1.0, 0.0], sds=[])

    def test_invalid_sd_non_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            Bandit.gaussian(sds=[1.0, 0.0])

    def test_reset_required(self):
        bandit = Bandit.gaussian()
        with pytest.raises(RuntimeError, match="Call reset"):
            bandit.sample_reward(0, 0)

    def test_protocol_compliance(self):
        assert isinstance(Bandit.gaussian(), TaskEnvironment)

    def test_n_arms_and_response_labels(self):
        bandit = Bandit.gaussian(
            means=[1.0, 0.0, -1.0],
            sds=[0.2, 0.4, 0.6],
            response_labels=[-1, 0, 1],
        )
        assert bandit.n_arms == 3
        assert bandit.response_labels == [-1, 0, 1]


class TestTaskConfig:
    def test_default_build(self):
        env = TaskConfig().build_environment()
        assert isinstance(env, Bandit)
        assert env.n_arms == 2

    def test_registered_tasks(self):
        assert "bandit" in registered_tasks()

    def test_builds_bernoulli(self):
        env = TaskConfig(
            task="bandit",
            reward="bernoulli",
            probabilities=[0.6, 0.4],
        ).build_environment()
        assert isinstance(env, Bandit)
        assert env.n_arms == 2

    def test_builds_gaussian(self):
        env = TaskConfig(
            task="bandit",
            reward="gaussian",
            means=[2.0, -1.0],
            sds=[0.5, 1.5],
        ).build_environment()
        assert isinstance(env, Bandit)
        assert env.n_arms == 2

    def test_custom_response_labels(self):
        env = TaskConfig(
            task="bandit",
            reward="bernoulli",
            probabilities=[0.5, 0.3, 0.2],
            response_labels=[10, 20, 30],
        ).build_environment()
        assert env.response_labels == [10, 20, 30]
        assert env.n_arms == 3

    def test_gaussian_custom_response_labels(self):
        env = TaskConfig(
            task="bandit",
            reward="gaussian",
            means=[2.0, 1.0, 0.0],
            sds=[0.2, 0.4, 0.6],
            response_labels=[10, 20, 30],
        ).build_environment()
        assert env.response_labels == [10, 20, 30]
        assert env.n_arms == 3

    def test_unknown_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            TaskConfig(task="maze").build_environment()

    def test_unknown_bandit_reward(self):
        with pytest.raises(ValueError, match="Unknown bandit reward"):
            TaskConfig(task="bandit", reward="exponential").build_environment()

    def test_unknown_option(self):
        with pytest.raises(TypeError, match="Unsupported options"):
            TaskConfig(
                task="bandit", reward="bernoulli", means=[1.0, 0.0]
            ).build_environment()
