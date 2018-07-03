from maze_env import Maze
from RL_brain import Sarsa_Lambda_Table

def update():
    for episode in range(100):
        observation = env.reset()
        action = RL.choose_action(str(observation))
        RL.eligibility_trace *= 0

        while True:
            env.render()
            observation_, reward, done = env.step(action)
            action_ = RL.choose_action(str(observation_))
            RL.learn(str(observation), action, reward, str(observation_), action_)
            action = action_
            observation = observation_
            if done:
                break
    print('game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = Sarsa_Lambda_Table(actions = list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()

