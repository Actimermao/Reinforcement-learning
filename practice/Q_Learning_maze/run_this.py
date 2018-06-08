
from maze_env import Maze
from RL_brain import QLearningTable
def update():
    pass
    for episode in range(100):
        #initial env
        observation = env.reset()
        while True:
            env.render()
            # use RL to choose action based on observation
            action = RL.choose_action(str(observation))

            #compute the reward and next observation base on the choosed action
            observation_, reward, done = env.step(action)

            #RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            observation = observation_

            #break while loop when end of this episode
            if done:
                break
    print("game over")
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
