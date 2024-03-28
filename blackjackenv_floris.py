import os
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.error import DependencyNotInstalled


def cmp(a, b):
    return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]

def can_split(hand): # Check if the player's hand can be split
    return len(hand) == 2 and hand[0] == hand[1]


class BlackjackEnv(gym.Env):
    """
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.

    ### Description
    Card Values:

    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-9) have a value equal to their number.

    This game is played with an infinite deck (or with replacement).
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards.

    The player can request additional cards (hit, action=1) until they decide to stop (stick, action=0)
    or exceed 21 (bust, immediate loss).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust, the player wins.
    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.

    ### Action Space
    There are four actions: stick (0), and hit (1), double down (2) and split (3).

    ### Observation Space
    The observation consists of a 3-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10 where 1 is ace),
    whether the player holds a usable ace (0 or 1),
    whether doubling down is allowed (0 or 1),
    whether splitting is allowed (0 or 1)


    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (http://incompleteideas.net/book/the-book-2nd.html).

    ### Rewards
    - win game: +1
    - lose game: -1
    - draw game: 0
    - win game with natural blackjack:

        +1.5 (if <a href="#nat">natural</a> is True)

        +1 (if <a href="#nat">natural</a> is False)

    ### Arguments

    ```
    gym.make('Blackjack-v1', natural=False, sab=False)
    ```

    <a id="nat">`natural=False`</a>: Whether to give an additional reward for
    starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).

    <a id="sab">`sab=False`</a>: Whether to follow the exact rules outlined in the book by
    Sutton and Barto. If `sab` is `True`, the keyword argument `natural` will be ignored.
    If the player achieves a natural blackjack and the dealer does not, the player
    will win (i.e. get a reward of +1). The reverse rule does not apply.
    If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).

    ### Version History
    * v0: Initial versions release (1.0.0)
    """

    def __init__(self, natural=False, sab=False):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), 
             spaces.Discrete(11), 
             spaces.Discrete(2),
             spaces.Discrete(2),
             spaces.Discrete(2))
        )

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        self.sab = sab

    def step(self, action):
        assert self.action_space.contains(action)
        
        if self.splitted:
            if self.active_hand == 1:
                observation, reward, terminated, _, _ = self._handle_action(action, self.player)
                
                if terminated:
                    # Switch to second hand if first hand done
                    self.active_hand = 2
                    terminated = False
                
                return observation, reward, terminated, False, {}
            elif self.active_hand == 2:
                observation, reward, terminated, _, _ = self._handle_action(action, self.player2)
                # Game is terminated if second hand is done
                return observation, reward, terminated, False, {}
        else:
            observation, reward, terminated, _, _ = self._handle_action(action, self.player)
            return observation, reward, terminated, False, {}

    def _handle_action(self, action, hand):
        if action == 1:  # hit: add a card to players hand and return
            hand.append(draw_card(self.np_random))
            
            if is_bust(hand):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
            self.can_double_down = False
        
        elif action == 0:  # stick: play out the dealers hand, and score
            terminated = True
            reward = 0.0 # klopt dit?
            
            if self.active_hand == 2 or not self.splitted:
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(draw_card(self.np_random))
                reward = cmp(score(self.player), score(self.dealer))
                

            if self.splitted:
                # Double reward for splitted hands
                reward *= 2
            
            # Natural blackjack only pays if no splitted hands
            elif self.sab and is_natural(hand) and not is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
            elif (
                not self.sab
                and self.natural
                and is_natural(hand)
                and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5
                
        elif action == 2: #double down: player gets one more card, game ends immediately with double reward
            # double down after splitting not allowed
            if len(hand) == 2 and not self.splitted:  
                hand.append(draw_card(self.np_random))  
                terminated = True  
                self.can_double_down = False

                if is_bust(hand):  
                    reward = -2.0 
                else:
                    while sum_hand(self.dealer) < 17: 
                        self.dealer.append(draw_card(self.np_random))
                    reward = 2 * cmp(score(hand), score(self.dealer))
                    
            else:
                raise ValueError("Double down not allowed")
        
        elif action == 3: #split: player splits the hand into two separate hands
            if can_split(hand) and not self.splitted:
                self.player, self.player2 = [self.player[0]], [self.player[1]]
                
                self.player.append(draw_card(self.np_random))
                self.player2.append(draw_card(self.np_random))
                
                self.splitted = True
                terminated = False
                reward = 0.0
                self.can_double_down = False
            else:
                raise ValueError("Split not allowed")
            
        return self._get_obs(), reward, terminated, False, {}
        
    def _get_obs(self):
        # first hand
        if self.active_hand == 1 or not self.splitted:
            return (sum_hand(self.player), 
                    self.dealer[0], 
                    usable_ace(self.player), 
                    self.can_double_down, 
                    self.splitted,
                    can_split(self.player))
        # second hand
        else:
            return (sum_hand(self.player2), 
                    self.dealer[0], 
                    usable_ace(self.player2), 
                    self.can_double_down, 
                    self.splitted,
                    False)
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        self.player2 = []
        self.can_double_down = True
        self.splitted = False
        self.active_hand = 1

        return self._get_obs(), {}