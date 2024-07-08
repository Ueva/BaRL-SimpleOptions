from typing import List
from numbers import Number


def discounted_return(rewards: List[Number], gamma: float) -> Number:
    """
    Given a list of rewards and a discount factor, computes the discounted sum of those rewards.

    Args:
        rewards (List[Number]): The list of rewards, where rewards[i] is the reward at time step i.
        gamma (float): The discount factor.

    Returns:
        Number: The discounted sum of rewards.

    Remarks:
        After testing many different variations of this function, I found that this
        pure-Python implementation is the fastest. Yes, faster than any reasonable NumPy equivalent.
        It is also the most readable. Do not try to "optimise" it without good reason.
    """
    discounted_sum_of_rewards = 0.0
    gamma_power = 1.0

    for reward in rewards:
        discounted_sum_of_rewards += reward * gamma_power
        gamma_power *= gamma

    return discounted_sum_of_rewards
