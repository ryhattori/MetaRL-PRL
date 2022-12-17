import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# %% Set training parameters
n_sessions = 300                # Number of training sessions
n_trials_per_session = 500      # Number of trials per session 
a2c_lr_rate = 0.0005            # Learning rate
a2c_gamma = 0.5                 # Discount factor, gamma
a2c_value = 0.01                # Coefficient for a value loss
a2c_entropy = 0.5               # Coefficient for entropy
a2c_unroll_trial = 10           # Unroll length
a2c_steps = 5                   # Time steps per trial
a2c_step_jitter = 1             # Time step jitter
a2c_ma_rate = 0                 # Miss rate

# Saving directory 
directory_name = r'C:\Users\Ryoma\Desktop\temp\temp'

keras.backend.clear_session()
sim_name = 'metarl_' + 'g' + str(a2c_gamma)[2:] + 'v' + str(a2c_value)[2:] + 'e' + str(a2c_entropy)[2:] + 'u' + str(a2c_unroll_trial) + 'l' + str(a2c_lr_rate)[2:] + 'ma' + str(a2c_ma_rate) + 's' + str(a2c_steps) + 'j' + str(a2c_step_jitter)
if not os.path.exists(os.path.join(directory_name, sim_name)):
    os.makedirs(os.path.join(directory_name, sim_name))


# %% Define functions
def _returns_advantages(rewards, values, next_value):
    returns = np.append(tf.zeros_like(rewards), next_value, axis=-1)
    for t in reversed(range(rewards.shape[0])):
        returns[t] = rewards[t] + a2c_gamma * returns[t + 1]
    returns = returns[:-1]
    advantages = returns - values
    return returns, advantages


@tf.function
def _value_loss(returns, value):
    return a2c_value * 0.5 * keras.losses.mean_squared_error(returns, value[:, -1])


@tf.function
def _logits_loss(acts_and_advs, logits):
    actions = tf.cast(acts_and_advs[:, :2], tf.float32)
    advantages = tf.cast(acts_and_advs[:, 2], tf.float32)
    weighted_ce = keras.losses.CategoricalCrossentropy(from_logits=True)
    policy_loss = weighted_ce(actions, logits[:, -1, :], sample_weight=advantages)
    probs = tf.nn.softmax(logits[:, -1, :])
    entropy_loss = keras.losses.categorical_crossentropy(probs, probs)
    ## alternative way to derive policy loss and entropy (same result)
    # policy = tf.nn.softmax(logits)
    # responsible_outputs = tf.reduce_sum(policy * actions, axis=1)
    # policy_loss = -tf.reduce_sum(tf.math.log(responsible_outputs + 1e-7) * advantages)
    # entropy_loss = -tf.reduce_sum(policy * tf.math.log(policy + 1e-7))
    return policy_loss - a2c_entropy * entropy_loss


# %% Build network
input_layer = keras.layers.Input(shape=(a2c_unroll_trial * a2c_steps, 3), dtype='float32')
hidden_logs = keras.layers.SimpleRNN(50, input_shape=[a2c_unroll_trial * a2c_steps, 3], dtype='float32', return_sequences=True)(input_layer)
hidden_vals = keras.layers.SimpleRNN(50, input_shape=[a2c_unroll_trial * a2c_steps, 3], dtype='float32', return_sequences=True)(input_layer)
logits_out = keras.layers.Dense(2, dtype='float32', name='policy_logits')(hidden_logs)
value_out = keras.layers.Dense(1, dtype='float32', name='value')(hidden_vals)
model = keras.Model(inputs=[input_layer], outputs=[logits_out, value_out])
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=a2c_lr_rate), loss=[_logits_loss, _value_loss])

model_actor = tf.keras.Model(model.inputs, model.layers[1].output)
model_critic = tf.keras.Model(model.inputs, model.layers[2].output)

# %% Define task and training conditions
class A2CAgent:
    def __init__(self, model, model_actor, model_critic):
        self.model = model
        self.model_actor = model_actor
        self.model_critic = model_critic

    def train(self, n_steps=5, step_jitter=1, unroll_trial_sz=10, n_trials_per_session=500, n_sessions=50, ma_rate=0):
        rnn_actor_activity = np.zeros((self.model.layers[1].units, n_steps * unroll_trial_sz, n_trials_per_session, n_sessions), dtype='float32')
        rnn_critic_activity = np.zeros((self.model.layers[2].units, n_steps * unroll_trial_sz, n_trials_per_session, n_sessions), dtype='float32')
        rewards, values, next_value, returns, advs = np.zeros((5, n_trials_per_session), dtype='float32')
        logits = np.zeros((n_trials_per_session, 2), dtype='float32')
        actions = np.zeros((n_trials_per_session, 2), dtype='float32')
        previous_action_rewards_seq = np.zeros((unroll_trial_sz * n_steps, 3), dtype='float32')
        previous_action_rewards = np.zeros((unroll_trial_sz, 3), dtype='float32')
        previous_action_rewards_unroll_batches_seq = np.zeros((n_trials_per_session, unroll_trial_sz * n_steps, 3), dtype='float32')
        previous_action_rewards_unroll_batches = np.zeros((n_trials_per_session, unroll_trial_sz, 3), dtype='float32')
 
        ep_rews = [0.0]
        prob_block = np.array([60, 80], dtype='int8')
        prob_set = np.array([[0.6, 0.1], [0.525, 0.175]], dtype='float32')
        trial_counter = 0
        session_counter = 0
        current_block = 0
        rew_assignment_history = np.zeros((n_trials_per_session, 2, n_sessions), dtype='int8')
        rew_history = np.zeros((n_trials_per_session, n_sessions), dtype='int8')
        action_history = np.zeros((n_trials_per_session, n_sessions), dtype='int8')
        value_history = np.zeros((n_trials_per_session, n_sessions), dtype='float32')
        logits_history = np.zeros((n_trials_per_session, 2, n_sessions), dtype='float32')
        return_history = np.zeros((n_trials_per_session, n_sessions), dtype='float32')
        rewprob_history = np.zeros((n_trials_per_session, 2, n_sessions), dtype='float32')
        for session_counter in range(n_sessions):
            trial_per_block = np.zeros(np.ceil(n_trials_per_session / 60).astype('int'), dtype='int16')
            for i in range(np.ceil(n_trials_per_session / 60).astype('int')):
                trial_per_block[i] = np.random.randint(prob_block[0], prob_block[1] + 1)
            trial_start_per_block = [0.] + [item + 1 for item in np.cumsum(trial_per_block) if item <= n_trials_per_session]
            trial_end_per_block = [item for item in np.cumsum(trial_per_block) if item <= n_trials_per_session] + [n_trials_per_session]
            num_blocks = len(trial_start_per_block)
            prob_set_list = np.zeros((num_blocks, 2), dtype='float32')
            if np.random.random() < 0.5:
                for i in range(num_blocks):
                    if np.mod(i, 4) == 0:
                        prob_set_list[i, :] = prob_set[0]
                    elif np.mod(i, 4) == 1:
                        prob_set_list[i, :] = np.flipud(prob_set[0])
                    if np.mod(i, 4) == 2:
                        prob_set_list[i, :] = prob_set[1]
                    elif np.mod(i, 4) == 3:
                        prob_set_list[i, :] = np.flipud(prob_set[1])
            else:
                for i in range(num_blocks):
                    if np.mod(i, 4) == 0:
                        prob_set_list[i, :] = np.flipud(prob_set[0])
                    elif np.mod(i, 4) == 1:
                        prob_set_list[i, :] = prob_set[0]
                    if np.mod(i, 4) == 2:
                        prob_set_list[i, :] = np.flipud(prob_set[1])
                    elif np.mod(i, 4) == 3:
                        prob_set_list[i, :] = prob_set[1]
            for i in range(2):
                rew_assignment_history[trial_counter, i, session_counter] = np.random.random() < prob_set_list[current_block, i]

            # Run trials
            for trial_counter in range(n_trials_per_session):
                temp = self.model.predict_on_batch(previous_action_rewards_seq[None, :, :])
                logits[trial_counter, :], values[trial_counter] = temp[0][0, -1, :], temp[1][0, -1, :]
                actions[trial_counter, :] = tf.one_hot(tf.squeeze(tf.random.categorical(logits[trial_counter, None, :], 1), axis=-1), 2)
                rnn_actor_activity[:, :, trial_counter, session_counter] = self.model_actor.predict_on_batch(previous_action_rewards_seq[None, :, :]).squeeze().T
                rnn_critic_activity[:, :, trial_counter, session_counter] = self.model_critic.predict_on_batch(previous_action_rewards_seq[None, :, :]).squeeze().T
                rewards[trial_counter] = rew_assignment_history[trial_counter, np.where(actions[trial_counter, :]), session_counter].astype('float32')
                if np.random.random() < ma_rate:
                    actions[trial_counter, :] = np.array([0, 0])
                    rewards[trial_counter] = np.array([0])

                iti = np.random.choice(np.arange(n_steps - step_jitter, n_steps + 0.01, 1, dtype='int'), 1)[0]
                previous_action_rewards_seq = np.insert(previous_action_rewards_seq[iti:, :], unroll_trial_sz * n_steps - iti, np.insert(np.zeros((iti - 1, 3)), 0, np.concatenate((actions[trial_counter, :], rewards[trial_counter, None]), axis=0), axis=0), axis=0)
                previous_action_rewards_unroll_batches_seq[trial_counter, :, :] = previous_action_rewards_seq

                previous_action_rewards = np.insert(previous_action_rewards, unroll_trial_sz, np.concatenate((actions[trial_counter, :], rewards[trial_counter, None]), axis=0), axis=0)[1:, :]
                previous_action_rewards_unroll_batches[trial_counter, :, :] = previous_action_rewards


                rew_history[trial_counter, session_counter] = rewards[trial_counter]
                if actions[trial_counter, 0] == 1:
                    action_history[trial_counter, session_counter] = 1
                elif actions[trial_counter, 1] == 1:
                    action_history[trial_counter, session_counter] = -1
                else:
                    action_history[trial_counter, session_counter] = 0

                value_history[trial_counter, session_counter] = values[trial_counter]
                logits_history[trial_counter, :, session_counter] = logits[trial_counter, :]
                rewprob_history[trial_counter, :, session_counter] = prob_set_list[current_block, :]

                ep_rews[-1] += rewards[trial_counter]
                if n_trials_per_session - 1 == trial_counter:
                    pass
                elif trial_end_per_block[current_block] == trial_counter + 1:
                    current_block += 1
                    for i in range(2):
                        if np.sum(actions[trial_counter, :]) != 0:
                            if rew_assignment_history[trial_counter, i, session_counter] == 1 and np.where(actions[trial_counter, :])[0][0] != i:
                                rew_assignment_history[trial_counter + 1, i, session_counter] = 1
                            else:
                                rew_assignment_history[trial_counter + 1, i, session_counter] = np.random.random() < prob_set_list[current_block, i]
                        else:
                            rew_assignment_history[trial_counter + 1, i, session_counter] = rew_assignment_history[trial_counter, i, session_counter]
                else:
                    for i in range(2):
                        if np.sum(actions[trial_counter, :]) != 0:
                            if rew_assignment_history[trial_counter, i, session_counter] == 1 and np.where(actions[trial_counter, :])[0][0] != i:
                                rew_assignment_history[trial_counter + 1, i, session_counter] = 1
                            else:
                                rew_assignment_history[trial_counter + 1, i, session_counter] = np.random.random() < prob_set_list[current_block, i]
                        else:
                            rew_assignment_history[trial_counter + 1, i, session_counter] = rew_assignment_history[trial_counter, i, session_counter]
            _, next_value = self.model.predict_on_batch(previous_action_rewards_seq[None, :, :])
            returns, advs = _returns_advantages(rewards, values, next_value[0, -1, 0].reshape(1))
            return_history[:, session_counter] = returns

            action_history_for_loss = action_history[unroll_trial_sz - 1:, session_counter]
            previous_action_rewards_unroll_batches_for_loss = previous_action_rewards_unroll_batches_seq[unroll_trial_sz - 1:, :, :]
            previous_action_rewards_unroll_batches_for_loss = previous_action_rewards_unroll_batches_for_loss[action_history_for_loss != 0, :, :]
            acts_and_advs = np.concatenate([actions, advs[:, None]], axis=-1)
            acts_and_advs_for_loss = acts_and_advs[unroll_trial_sz - 1:, :]
            acts_and_advs_for_loss = acts_and_advs_for_loss[action_history_for_loss != 0, :]
            returns_for_loss = returns[unroll_trial_sz - 1:]
            returns_for_loss = returns_for_loss[action_history_for_loss != 0]

            # Train the network
            losses = self.model.train_on_batch(tf.convert_to_tensor(previous_action_rewards_unroll_batches_for_loss), [acts_and_advs_for_loss, returns_for_loss])

            logging.debug("[%d/%d] Losses: %s" % (session_counter + 1, n_sessions, losses))
            previous_action_rewards_seq = np.zeros((unroll_trial_sz * n_steps, 3), dtype='float32')
            previous_action_rewards = np.zeros((unroll_trial_sz, 3), dtype='float32')
            previous_action_rewards_unroll_batches_seq = np.zeros((n_trials_per_session, unroll_trial_sz * n_steps, 3), dtype='float32')
            previous_action_rewards_unroll_batches = np.zeros((n_trials_per_session, unroll_trial_sz, 3), dtype='float32')
            current_block = 0
            ep_rews.append(0.0)
            if session_counter < n_sessions:
                for i in range(2):
                    rew_assignment_history[trial_counter, i, session_counter] = np.random.random() < prob_set_list[current_block, i]
            logging.info("Episode: %03d, Reward rate: %03f" % (len(ep_rews) - 1, ep_rews[-2] / n_trials_per_session))
        return rew_history, action_history, value_history, return_history, logits_history, rew_assignment_history, rewprob_history, rnn_actor_activity, rnn_critic_activity

# %% Train netowrk
logging.getLogger().setLevel(logging.INFO)
agent = A2CAgent(model, model_actor, model_critic)

rew_assignment_history = np.zeros((n_trials_per_session, 2, n_sessions), dtype='int8')
rew_history = np.zeros((n_trials_per_session, n_sessions), dtype='int8')
action_history = np.zeros((n_trials_per_session, n_sessions), dtype='int8')
value_history = np.zeros((n_trials_per_session, n_sessions), dtype='float32')
logits_history = np.zeros((n_trials_per_session, 2, n_sessions), dtype='float32')
return_history = np.zeros((n_trials_per_session, n_sessions), dtype='float32')
rewprob_history = np.zeros((n_trials_per_session, 2, n_sessions), dtype='float32')
rnn_actor_activity = np.zeros((50, a2c_steps * a2c_unroll_trial, n_trials_per_session, n_sessions), dtype='float32')
rnn_critic_activity = np.zeros((50, a2c_steps * a2c_unroll_trial, n_trials_per_session, n_sessions), dtype='float32')

#  Training
for session_id in range(n_sessions):
    print('Running ' + str(session_id + 1) + '/' + str(n_sessions) + ' episode...')
    rew_history[:, session_id, None], action_history[:, session_id, None], value_history[:, session_id, None], \
    return_history[:, session_id, None], logits_history[:, :, session_id, None], rew_assignment_history[:, :, session_id, None], rewprob_history[:, :, session_id, None], \
    rnn_actor_activity[:, :, :, 0, None], rnn_critic_activity[:, :, :, 0, None] \
        = agent.train(unroll_trial_sz=a2c_unroll_trial, n_trials_per_session=n_trials_per_session, n_sessions=1, ma_rate=a2c_ma_rate)
    agent.model.save_weights(os.path.join(directory_name, sim_name, 'weight_' + str(session_id + 1)))

print("Finished training.")

# %% Save behavior and activity data
np.savez_compressed(os.path.join(directory_name, sim_name, sim_name + '_history'),
            rew_history=rew_history, action_history=action_history, value_history=value_history,
            return_history=return_history, logits_history=logits_history, rew_assignment_history=rew_assignment_history, rewprob_history=rewprob_history,
            rnn_actor_activity=rnn_actor_activity, rnn_critic_activity=rnn_critic_activity
            )

# %% Analyze the behaviors during training
Rc = np.zeros((rew_history.shape[0] - 5, rew_history.shape[1]), dtype='int8')
Rp1 = np.zeros((rew_history.shape[0] - 5, rew_history.shape[1]), dtype='int8')
Rp2 = np.zeros((rew_history.shape[0] - 5, rew_history.shape[1]), dtype='int8')
Rp3 = np.zeros((rew_history.shape[0] - 5, rew_history.shape[1]), dtype='int8')
Rp4 = np.zeros((rew_history.shape[0] - 5, rew_history.shape[1]), dtype='int8')
Rp5 = np.zeros((rew_history.shape[0] - 5, rew_history.shape[1]), dtype='int8')
Cc = np.zeros((rew_history.shape[0] - 5, rew_history.shape[1]), dtype='int8')
Cp1 = np.zeros((rew_history.shape[0] - 5, rew_history.shape[1]), dtype='int8')
Cp2 = np.zeros((rew_history.shape[0] - 5, rew_history.shape[1]), dtype='int8')
Cp3 = np.zeros((rew_history.shape[0] - 5, rew_history.shape[1]), dtype='int8')
Cp4 = np.zeros((rew_history.shape[0] - 5, rew_history.shape[1]), dtype='int8')
Cp5 = np.zeros((rew_history.shape[0] - 5, rew_history.shape[1]), dtype='int8')
for ep_id in range(Rc.shape[1]):
    for trial_id in range(Rc.shape[0]):
        Rc[trial_id, ep_id] = 2 * rew_history[trial_id + 5, ep_id] - 1
        Rp1[trial_id, ep_id] = 2 * rew_history[trial_id + 4, ep_id] - 1
        Rp2[trial_id, ep_id] = 2 * rew_history[trial_id + 3, ep_id] - 1
        Rp3[trial_id, ep_id] = 2 * rew_history[trial_id + 2, ep_id] - 1
        Rp4[trial_id, ep_id] = 2 * rew_history[trial_id + 1, ep_id] - 1
        Rp5[trial_id, ep_id] = 2 * rew_history[trial_id, ep_id] - 1
        Cc[trial_id, ep_id] = action_history[trial_id + 5, ep_id]
        Cp1[trial_id, ep_id] = action_history[trial_id + 4, ep_id]
        Cp2[trial_id, ep_id] = action_history[trial_id + 3, ep_id]
        Cp3[trial_id, ep_id] = action_history[trial_id + 2, ep_id]
        Cp4[trial_id, ep_id] = action_history[trial_id + 1, ep_id]
        Cp5[trial_id, ep_id] = action_history[trial_id, ep_id]
RewCc = Cc.copy()
RewCc[Rc == -1] = 0
UnrCc = Cc.copy()
UnrCc[Rc == 1] = 0
RewCp1 = Cp1.copy()
RewCp1[Rp1 == -1] = 0
UnrCp1 = Cp1.copy()
UnrCp1[Rp1 == 1] = 0
RewCp2 = Cp2.copy()
RewCp2[Rp2 == -1] = 0
UnrCp2 = Cp2.copy()
UnrCp2[Rp2 == 1] = 0
RewCp3 = Cp3.copy()
RewCp3[Rp3 == -1] = 0
UnrCp3 = Cp3.copy()
UnrCp3[Rp3 == 1] = 0
RewCp4 = Cp4.copy()
RewCp4[Rp4 == -1] = 0
UnrCp4 = Cp4.copy()
UnrCp4[Rp4 == 1] = 0
RewCp5 = Cp5.copy()
RewCp5[Rp5 == -1] = 0
UnrCp5 = Cp5.copy()
UnrCp5[Rp5 == 1] = 0

log_reg = LogisticRegression(solver='lbfgs', penalty='none', n_jobs=-1, multi_class='auto', fit_intercept=True, max_iter=10000)
predictors = np.concatenate((RewCp1[:, np.newaxis, :], RewCp2[:, np.newaxis, :], RewCp3[:, np.newaxis, :],
                                RewCp4[:, np.newaxis, :], RewCp5[:, np.newaxis, :],
                                Cp1[:, np.newaxis, :], Cp2[:, np.newaxis, :], Cp3[:, np.newaxis, :],
                                Cp4[:, np.newaxis, :], Cp5[:, np.newaxis, :]), axis=1)
cv_fold = 10
kf1 = KFold(n_splits=cv_fold, shuffle=False)
cv_ind = 0
coef_norm = np.zeros((predictors.shape[1] + 1, predictors.shape[2]))
coef = np.zeros((predictors.shape[1] + 1, predictors.shape[2]))
acc = np.zeros(predictors.shape[2])
for session_id in range(predictors.shape[2]):
    true_class_list = np.array([])
    predict_class_list = np.array([])
    try:
        choice_id = (Cc[:, session_id] == 1) | (Cc[:, session_id] == -1)
        Cc_woma = Cc[choice_id, session_id]
        predictors_woma = predictors[choice_id, :, session_id]
        for train_index, test_index in kf1.split(Cc_woma):
            log_reg.fit(predictors_woma[train_index, :], Cc_woma[train_index])
            true_class_list = np.append(true_class_list, Cc_woma[test_index])
            predict_class_list = np.append(predict_class_list, log_reg.predict(predictors_woma[test_index, :]))
            cv_ind = cv_ind + 1
        acc[session_id] = np.mean((predict_class_list * true_class_list) == 1)
        fitall_model = log_reg.fit(predictors_woma, Cc_woma)
        coef[:, session_id] = np.concatenate((fitall_model.coef_[-1, :], fitall_model.intercept_))
        coef_norm = np.zeros_like(coef)
        for ep_id in range(len(acc)):
            if acc[ep_id] > 0.5:
                coef_norm[:, ep_id] = (acc[ep_id] - 0.5) * (coef[:, ep_id] / (np.sum(abs(coef[:, ep_id]), axis=0) + 10 ** (-10)))
            else:
                coef_norm[:, ep_id] = 1e-10 * (coef[:, ep_id] / (np.sum(abs(coef[:, ep_id]), axis=0) + 10 ** (-10)))
    except:
        print(str(session_id) + '_skipped')
np.savez_compressed(os.path.join(directory_name, sim_name, 'coef_norm'),
                    coef_norm=coef_norm,
                    coef=coef,
                    acc=acc,
                    )

trials_per_split = 50
fig, axs = plt.subplots(3, 3, figsize=(12, 7.5), tight_layout=True)
sns.set_style("ticks")
sns.despine()
ax = axs[0, 0]
ax.plot(np.arange(0, rew_history.shape[1], 1), np.mean(rew_history, axis=0))
ax.set_xlabel('Episode')
ax.set_ylabel('Reward rate')
ax = axs[0, 1]
ax.plot(np.arange(0, rew_history.shape[1], 1), np.mean(np.sum(rew_assignment_history, axis=1) != 0, axis=0))
ax.set_xlabel('Episode')
ax.set_ylabel('Reward availability rate')
ax = axs[0, 2]
ax.plot(np.arange(0, rew_history.shape[1], 1), np.mean(rew_history, axis=0) / np.mean(np.sum(rew_assignment_history, axis=1) != 0, axis=0))
ax.set_xlabel('Episode')
ax.set_ylabel('Harvesting rate')
ax = axs[1, 0]
ax.plot(np.arange(0, rew_history.shape[1], 1), np.sum(coef_norm[:5, :], axis=0))
ax.axhline(y=0, color='k', linestyle=':')
ax.set_xlabel('Episode')
ax.set_ylabel('$\Sigma$RewC')
ax = axs[1, 1]
ax.plot(np.arange(0, rew_history.shape[1], 1), np.sum(coef_norm[5:10, :], axis=0))
ax.axhline(y=0, color='k', linestyle=':')
ax.set_xlabel('Episode')
ax.set_ylabel('$\Sigma$C')
ax = axs[2, 0]
ax.axhline(y=0, color='k', linestyle=':')
n_split = np.floor(coef_norm.shape[1] / trials_per_split).astype('int')
for i in range(n_split):
    ax.plot(np.arange(-1, -6, -1), np.mean(coef_norm[:5, i * trials_per_split + 1:(i + 1) * trials_per_split + 1], axis=1), c=cm.get_cmap('turbo')(i / (n_split - 1)),)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.get_cmap('turbo')), ax=ax, ticks=[0, 1])
cbar.set_ticks(np.arange(0, 1.01, 1 / (n_split - 1)))
cbar.ax.set_yticklabels(np.arange(trials_per_split / 2, coef_norm.shape[1] + 0.001, (coef_norm.shape[1] - np.mod(coef_norm.shape[1], trials_per_split)) / n_split))
ax.set_xticks(np.arange(-1, -6, -1))
ax.set_xlabel("Past Trials", fontsize=13)
ax.set_ylabel("Normalized weight", fontsize=13)
ax = axs[2, 1]
ax.axhline(y=0, color='k', linestyle=':')
n_split = np.floor(coef_norm.shape[1] / trials_per_split).astype('int')
for i in range(n_split):
    ax.plot(np.arange(-1, -6, -1), np.mean(coef_norm[5:10, i * trials_per_split + 1:(i + 1) * trials_per_split + 1], axis=1), c=cm.get_cmap('turbo')(i / (n_split - 1)))
cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.get_cmap('turbo')), ax=ax, ticks=[0, 1])
cbar.set_ticks(np.arange(0, 1.01, 1 / (n_split - 1)))
cbar.ax.set_yticklabels(np.arange(trials_per_split / 2, coef_norm.shape[1] + 0.001, (coef_norm.shape[1] - np.mod(coef_norm.shape[1], trials_per_split)) / n_split))
ax.set_xticks(np.arange(-1, -6, -1))
ax.set_xlabel("Past Trials", fontsize=13)
ax.set_ylabel("Normalized weight", fontsize=13)
fig.show()
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig(os.path.join(directory_name, sim_name, sim_name + '_AcrossEp.svg'), format="svg", dpi=300)
plt.close()