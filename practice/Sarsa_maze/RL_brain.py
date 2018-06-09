import numpy as np
import pandas  as pd

class RL():
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.actions,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() > self.epsilon:
            action = np.random.choice(self.actions)

        else:
            state_action = self.q_table.loc[observation,:]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()

        return action




    def learn(self, *args):
        pass

class QlearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super().__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)

        prediction = self.q_table[s, a]
        if s_  == 'terminal':
            target = r
        else:
            target = r + self.gamma*self.q_table.loc[s_, :].max()

        self.q_table[s, a] += self.lr * (target-prediction)


class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super().__init__(actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9)


    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        prediction = self.q_table.loc[s, a]
        if s_ == 'terminal':
            target = r
        else:
            target = r + self.gamma*self.q_table.loc[s_, a_]
        self.q_table.loc[s,a] += self.lr*(target - prediction)
