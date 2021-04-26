agent_params = {
    'hidden_layer_dim':32,
    'batch_size': 64,
    'replay_buffer_capacity': 10_000,
    'gamma': 0.9,
    'epsilon_start': 0.99,
    'epsilon_end': 0.05,
    'epsilon_decay_rate': 5_000,
    'burnin_steps': 1_000,
    'soft_update': True,
    'tau': 0.005,
}

vdn_params = {
    'steps_per_episode': 50,
    'hidden_layer_dim': 32,
    'loss_func': 'MSE',
    'lr': 0.001,
    'clip_val': 1.,
}