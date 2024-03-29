from blackjack_agent import BlackjackAgent

class BasicStrategyAgent(BlackjackAgent):
    def __init__(self, env):
        super().__init__(env)
        
    def minimum_standing_number(self, D, usable_ace):
        # Draw card if player sum is less than minimum standing number
        if usable_ace:
            if D <= 8 or D == 1:
                return 18
            elif D == 9 or D == 10:
                return 19
        else:
            if D in [2,3]:
                return 13
            elif D in [4,5,6]:
                return 12
            elif D >= 7 or D == 1:
                return 17
        
        
    def action_selector(self, observation):
        player_sum = observation[0]
        dealer_card = observation[1]
        usable_ace = observation[2]
        double_down_allowed = observation[3]
        splitted = observation[4]
        can_split = observation[5]
        
        # Strategy implemented according to Baldwin et al. (1956)
        
        if player_sum < self.minimum_standing_number(dealer_card, usable_ace):
            action = 1
        else: 
            action = 0
        
        if double_down_allowed:
            if usable_ace:
                if ((player_sum == 18 and dealer_card in [4,5,6])
                or (player_sum == 17 and dealer_card in [3,4,5,6])
                or (player_sum in [13,14,15,16] and dealer_card in [5,6])
                or (player_sum == 12 and dealer_card == 5)):
                    action = 2
            else:
                if ((player_sum == 11 and dealer_card >= 2 and dealer_card <= 10)
                or (player_sum == 10 and dealer_card >= 2 and dealer_card <= 9)
                or (player_sum == 9 and dealer_card >= 2 and dealer_card <= 6)):
                    action = 2
        
        if not splitted and can_split:
            if (player_sum == 16
            or player_sum == 12
            or (player_sum == 18 and dealer_card in [2,3,4,5,6,8,9])
            or (player_sum == 14 and dealer_card >= 2 and dealer_card <= 8)
            or (player_sum in [4, 6, 12] and dealer_card >= 2 and dealer_card <= 7)
            or (player_sum == 8 and dealer_card == 5)):
                action = 3
        return action