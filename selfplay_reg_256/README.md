# Selfplay Baseline Agent

**Agent name:** SelfplayAgent

**Author (s):** Rohit Senthil (rsenthil8@gatech.edu)

## Description

An agent trained with PPO via multi-agent self-play using Ray RLLib. See the `ray_selfplay.py` file for more details on the training regime. Model parameters are:

```python
"vf_share_layers": False,
"fcnet_hiddens": [256, 256],
"fcnet_activation": "relu",
```

Trained with the default observation space.
