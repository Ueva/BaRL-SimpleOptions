import pytest

from barl_simpleoptions import OptionAgent

# Test single reward.
def test_discounted_reward_single():
    rewards = [1]
    gamma = 0.9
    correct_discounted_reward = 1

    discounted_reward = OptionAgent(None)._discounted_return(rewards, gamma)

    assert discounted_reward == correct_discounted_reward


# Test multiple rewards.
def test_discounted_reward_multiple():
    rewards = [1, 2, 3, 4, 5]
    gamma = 0.9
    correct_discounted_reward = 1 * gamma ** 0 + 2 * gamma ** 1 + 3 * gamma ** 2 + 4 * gamma ** 3 + 5 * gamma ** 4

    discounted_reward = OptionAgent(None)._discounted_return(rewards, gamma)

    assert discounted_reward == correct_discounted_reward


# Test for zero total reward.
def test_discounted_reward_zero():
    rewards = [0, 0, 0]
    gamma = 0.9
    correct_discounted_reward = 0

    discounted_reward = OptionAgent(None)._discounted_return(rewards, gamma)

    assert discounted_reward == correct_discounted_reward


# Test for negative rewwards.
def test_discounted_reward_negative():
    rewards = [-1, -2, -3]
    gamma = 0.9
    correct_discounted_reward = -1 * gamma ** 0 + -2 * gamma ** 1 + -3 * gamma ** 2

    discounted_reward = OptionAgent(None)._discounted_return(rewards, gamma)

    assert discounted_reward == correct_discounted_reward