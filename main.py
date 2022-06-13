import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter

# Custom class for implementing TD3 Critic and Actor
from TD3 import TD3
# Custom class for implementing Replay Buffer to support Actor and Critc pair
import ReplayBuffer

# For rendoring the result through Ubuntu on Cloud environment
# Remark: please comment the following line if you are using local GPU resources
os.environ["SDL_VIDEODRIVER"] = "x11"

# Check if CUDA can be found for GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(seed):
    # In our case, there are concrete terminal states for our BipedalWalker
    # Thus, we flag it as True by default
    has_terminal_state = True

    # Prepare the environment
    # Remark: There is a typo in https://github.com/openai/gym/wiki/BipedalWalker-v2
    #         Although it is named as v2, but it is actually v3
    # BipedalWalker-v3 is a challenging environment in the Gym
    # Our agent should run very fast, should not trip himself off, should use as little energy as possible

    env = gym.make('BipedalWalkerHardcore-v3')
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    act_noise = 0.25
    print('  state_space:', state_space, '  action_space:', action_space, '  max_a:', max_action, '  min_a:', env.action_space.low[0])

    # If you are going to train the model from scratch, kindly set both is_rendor_model and is_load_model flags to `False`
    # Remark: highly recommend to train your own with GPU resources, 
    #         or else you will experience a long training time
    is_rendor_model = True
    is_load_model = True

    # The Actor and Critic pair being trained are named as `ppo_actor{n}.pth` and `ppo_q_critic{n}.pth` correspondingly
    # You can change n by setting model_index to a different value
    model_index = 3600
    
    random_seed = seed

    # Kindly specify the no. of episode desired, we comment our setting as below line
    #max_episode = 3600
    max_episode = 2000000
    # Set an episode interval to save models
    save_interval = 400 

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    # Log related events for debugging
    writer = SummaryWriter(log_dir='runs/exp')

    kwargs = {
        "has_terminal_state": has_terminal_state,
        "state_space": state_space,
        "action_space": action_space,
        "max_action": max_action,
        # Discount factor. (Always between 0 and 1.)
        "gamma": 0.99,
        "net_width": 200,
        # Learning rate for policy - actor network
        "a_lr": 1e-4,
        # Learning rate for Q-networks - Critic network
        "c_lr": 1e-4,
        # Minibatch size for SGD
        "q_batchsize":256,
    }

    # Pass the agruments to TD3.py for training preparation
    model = TD3(**kwargs)
    # If it is not training, the model_index is passed for loading the saved models
    if is_load_model: model.load(model_index)
    # Initialize the replay buffer
    # Remarks: For the ease of development, we hardcode the replay_size (int) – Maximum length of replay buffer
    replay_buffer = ReplayBuffer.ReplayBuffer(state_space, action_space, max_size=int(1e6))

    # Initalize an array to save all episode rewards for further processing
    all_episode_rews = []

    for episode in range(max_episode):
        s, done = env.reset(), False
        episode_rew = 0
        # Number of steps of interaction (state-action pairs) for the agent and the environment in each epoch
        steps = 0
        # Stddev for Gaussian exploration noise added to policy at training time
        act_noise *= 0.999

        # Loop the interaction until terminal state is reached
        while not done:
            steps += 1
            # Interaction based on trained models
            if is_rendor_model:
                a = model.select_action(s)
                s_prime, r, done, info = env.step(a)
                env.render()
            # Here comes with training part
            else:
                # Select the optimal action based on the limited action range / space
                # abbreviations used in the below training part
                # a = action
                # s = current state
                # r = reward
                # s_prime = target state

                a = ( model.select_action(s) + np.random.normal(0, max_action * act_noise, size=action_space)
                    ).clip(-max_action, max_action)
                s_prime, r, done, info = env.step(a)

                # Tricks for BipedalWalker
                # If accident is happened, we reset r=-1 and mark it as `dead` 
                if r <= -100:
                    r = -1
                    replay_buffer.add(s, a, r, s_prime, True)
                else:
                    replay_buffer.add(s, a, r, s_prime, False)

                # If the replay buffer reaches 2000 rows, we train the model by using replay buffer
                if replay_buffer.size > 2000: model.train(replay_buffer)

            s = s_prime
            episode_rew += r

        # Save the model according to the interval set
        # We soley use the plotting part for debugging or evaluation purpose
        if (episode+1)%save_interval==0:
            model.save(episode + 1)
            # plt.plot(all_episode_rews)
            # plt.savefig('seed{}-ep{}.png'.format(random_seed,episode+1))
            # plt.clf()


        # Record the model result and print system logs to console
        # If you are using remote / cloud server, kindly ensure the `fake screen` is enabled for the printing
        if episode == 0: all_episode_rews.append(episode_rew)
        else: all_episode_rews.append(all_episode_rews[-1]*0.9 + episode_rew*0.1)
        writer.add_scalar('sum_of_episode_rews', all_episode_rews[-1], global_step=episode)
        writer.add_scalar('episode_rew', episode_rew, global_step=episode)
        writer.add_scalar('exploration', act_noise, global_step=episode)
        print('seed:',random_seed,'episode:', episode,'score:', episode_rew, 'step:',steps , 'max:', max(all_episode_rews))

    env.close()



if __name__ == '__main__':
    main(seed=1)


