import numpy as np
import time
import matplotlib.pyplot as plt
import os
from tasks.task import get_task
from analyzer import Q
#from matplotlib import animation

def display_frames_as_gif(frames, task):
    m = __import__('matplotlib.animation')
    animation = m.animation
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=5)
    anim.save(task + '.gif', writer='imagemagick', fps=30)

def eval(batches, render=False):
    simTime = time.time()
    pop, env, seeds, debug, config = batches
    fit = []
    pFit = []
    logs = []
    result = {}
    totalSteps = 0
    frames = []
    all_angles = []
    all_actions = []
    values = []
    pop.compile()
    nn = pop.execute()
    nn.compile()

    for seed in seeds:
        env.seed(seed=seed)
        obs = env.reset()
        while not isinstance(obs, np.ndarray):
            obs = obs[0]
        done = False
        totalReward = 0
        step = 0
        joint_angles = []
        actions = []
        if (debug):
            log = ''
        while (not done):
            if (env.__class__.__name__ in ('CartPole', 'Retina', 'RetinaNM', 'Parity')):
                action, stats = nn.step(obs)
                if (action > 0):
                    action = np.array([[1]])
                else:
                    action = np.array([[0]])
            else:
                action, stats = nn.step(obs)
                action *= env.action_space.high

            if (env.__class__.__name__ in ('TimeLimit')):
                action = action.flatten()
            elif (step % config.ns_interval == 0):
                #actions.append(action.copy())
                actions.append(action)
            out = env.step(action)
            newObs, reward, done, info = out[:4]
            totalReward += reward
            totalSteps += 1
            if totalSteps > 10000:
                done = True
            if (debug):
                steplog = 'STEP: {0} \n INPUT: {1} \n OUTPUT: {2} \n NN VALUES: {3} \n REWARD: {4} \n'.format(
                    step, obs, action, nn.values, totalReward)
                log += steplog
                step += 1
            if (env.__class__.__name__ in ['BipedalWalker', 'BipedalWalkerHardcore']):
                joint_angles.append(newObs[4] - newObs[9])
                if (max(abs(obs - newObs)) < 1e-5):
                    done = True
            obs = newObs
            if render:
                frames.append(env.render(mode = 'rgb_array'))

            if 'neuron_values' in stats.keys():
                values.append(stats['neuron_values'])

        if (env.__class__.__name__ in ('TimeLimit')):
            actions, values = gym_sample_batch(nn, env)
        all_actions.extend(actions)
        fit.append(totalReward)
        all_angles.append(joint_angles)
        if totalReward > 0:
            pFit.append(totalReward - nn.get_penalty())
        else:
            pFit.append(totalReward)
        if (debug):
            logs.append(log)
        env.close()
    pop.detach()
    simTime = time.time() - simTime
    result['fit'] = np.mean(fit)
    pop.fitness = result['fit']
    result['pfit'] = np.mean(pFit)
    if (debug):
        result['logs'] = logs
    result['pop'] = pop
    #q, groups = Q(nn)
    #result['Q'] = q
    result['steps'] = totalSteps
    result['time'] = simTime
    result['task'] = env.__class__.__name__
    result['seeds'] = seeds
    result['neuron_values'] = values
    result['evals'] = 1
    if (config.behavior_similarity):
        all_actions = np.array(all_actions)
        if len(all_actions.shape) > 2:
            all_actions = all_actions.reshape((-1,all_actions.shape[2]))
        result['actions'] = all_actions
    if (env.__class__.__name__ in ['BipedalWalker', 'BipedalWalkerHardcore']):
        mean, std = angles_statics(all_angles)
        result['me_features'] = (mean, std)
    if render:
        display_frames_as_gif(frames, env.__class__.__name__)
    return result

def gym_sample_batch(nn, env, seed=1234, sample_size = 1000):
    generator = np.random.default_rng(seed)
    batch = []
    for i in range(sample_size):
        batch.append(generator.uniform(env.observation_space.low, env.observation_space.high))
    actions = []
    values = []
    for row in batch:
        action, stats = nn.step(row)
        actions.append(action)
        if 'neuron_values' in stats.keys():
            values.append(stats['neuron_values'])
    return actions, values

def angles_statics(all_angles):
    n = []
    sums = []
    diff = 0.0
    for angles in all_angles:
        n.append(len(angles))
        mean = np.mean(angles)
        for angle in angles:
            diff += (mean - angle)**2
        sums.append(abs(mean) * len(angles))
    std = (diff / (sum(n) + 0.0001)) ** 0.5
    mean = sum(sums) / (sum(n) + 0.0001)
    return mean, std

class EvalSummary():
    def __init__(self):
        self.store = {}
        self.store_id = {}
        self.metricName = ''
        self.summarized_keys = ['best', 'longest', 'bestFit', 'evals']

    def reduce(self, result, metricName):
        for d in result:
            self.store_id[d['pop'].ID] = d
        self.metricName = metricName
        keys = result[0].keys()
        for key in keys:
            self.store[key] = []
            for d in result:
                self.store[key].append(d[key])
        metrics = self.store[metricName]
        indices = np.flip(np.argsort(metrics)).tolist()
        for key in self.store.keys():
            arr = self.store[key]
            self.store[key] = [arr[i] for i in indices]
        self.evals = sum(self.store['evals'])
        self.store['evals'] = self.evals
        #print(self.evals)

    def get_metric(self):
        return self.store[self.metricName]

    def get_res(self, key):
        return self.store[key]

    def get_res_by_ID(self, ID, key):
        return self.store_id[ID][key]

    def get_summary(self):
        return {k:v for k,v in self.store.items() if k in self.summarized_keys}

    def add_summary_key(self, key):
        self.summarized_keys.append(key)
    def set_summary(self, key, value):
        self.store[key] = value

    def summarize(self):
        self.store['bestFit'] = max(self.store['fit'])
        self.store['best'] = self.store['pop'][0]
        #index = np.argmax(self.store['time'])
        #self.store['longest'] = (index, self.store['pop'][index])
        try:
            del self.store['pop']
            del self.store['actions']
            del self.store['neuron_values']
        except KeyError:
            pass
