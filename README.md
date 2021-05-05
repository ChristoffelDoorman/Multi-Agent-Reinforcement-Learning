# Multi-Agent-Reinforcement-Learning
Solving the [ma-gym](https://github.com/koulanurag/ma-gym) (multi-agent version of [OpenAI gym](https://github.com/openai/gym)) game Switch with 2 and 4 agents using DQN (PPO will be added soon).

Key for optimal performance of the 4-agent version of Switch is that the DQN agents use a [Value Decomposition Network](https://arxiv.org/pdf/1706.05296.pdf) and soft-updates of the target network.
