# Towards Cost Sensitive Decision Making

## Usage
```bash
python scripts/run_agent.py --cfg_file=path/to/config --mode=train/test
```

## Code Structure
    .
    |- scripts
    |- src
    |   |- agents
    |   |   |- fully_observed_ppo: fully observed
    |   |   |- concat_action_ppo: concatenate afa and tsk action space
    |   |   |- concat_action_mbppo: with generative model
    |   |   |- batch_hier_ppo: batch acquisition
    |   |   |- batch_hier_mbppo: with generative model
    |   |   |- seque_hier_ppo: sequential acquisition
    |   |   |- seque_hier_mbppo: with generative model
    |   |- environemnts
    |   |   |- sepsis: sepsis simulator
    |   |   |- episode_length_wrapper: limit episode length
    |   |   |- concat_action_wrapper: concat action space
    |   |   |- batch_acquire_env: batch acquisition
    |   |   |- seque_acquire_env: sequential acquisition
    |   |- models
    |   |   |- poex_vae_cat_dis: POEx model for environments with categorical observations and discrete actions
    |   |- networks
    |   |- policies
    |   |- utils
    |- requirements.txt
    |- README.md