from env import Robot_Gridworld
from deep_q_learning import DeepQLearning
import matplotlib.pyplot as plt
import numpy as np

returns = []
gamma = 0.99
def update():
    step = 0

    for episode in range(300):

        state = env.reset()
        step_count = 0
        episode_return = 0

        while True:

            env.render()
            action = dqn.choose_action(state)
            next_state, reward, terminal = env.step(action)


            step_count += 1
            dqn.store_transition(state, action, reward, next_state)

            if (step > 200) and (step % 5 == 0):
                dqn.learn()
            #### Begin learning after accumulating certain amount of memory #####
            state = next_state
            episode_return += reward

            if terminal == True:
                print(" {} End. Total steps : {}\n".format(episode + 1, step_count))
                break

            step += 1
            returns.append(episode_return)

   ############# Implement the codes to plot 'returns per episode' ####################
   ############# You don't need to place your plotting code right here ################
   

    print('Game over.\n')
    env.destroy()
    
    # return returns


if __name__ == "__main__":

    env = Robot_Gridworld()

    ###### Recommended hyper-parameters. You can change if you want ###############
    print(env.n_actions, env.n_features)
    dqn = DeepQLearning(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        discount_factor=0.9,
                        e_greedy=0.05,
                        replace_target_iter=200,
                        memory_size=2000
                        )


    env.after(100, update) #Basic module in tkinter
    env.mainloop() #Basic module in tkinter
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(returns)),returns)
    plt.xlabel("Episodes")
    plt.ylabel("Return per Episode")
    plt.legend()
    plt.show()

