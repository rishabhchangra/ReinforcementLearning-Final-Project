import gym
import numpy as np
from ddqn_agent import DDQNAgent
from gym import wrappers
from utils import plot_learning_curve, make_env

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    #to test change the load_checkpoint to True
    load_checkpoint = False
    n_games = 250
    agent = DDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=10000, eps_min=0.1,
                     batch_size=32, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DDQNAgent',
                     env_name='PongNoFrameskip-v4', test=load_checkpoint)

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

  
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()
        # print("after reset")

        score = 0
       
        while not done:
            # env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

           
            # print("After main step" )
            # if score==20:
            #     print(reward)
            
            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, int(done))
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        if load_checkpoint and n_steps >= 18000:
            break


    plot_learning_curve(steps_array, scores, eps_history, figure_file)
