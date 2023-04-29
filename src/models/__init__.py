def get_model(config, obs_space, act_space):
    if config.name == "poex_cat_dis":
        from .poex_vae_cat_dis import POExVAE
        model = POExVAE(config, obs_space, act_space)
    elif config.name == "poex_con_dis":
        from .poex_vae_con_dis import POExVAE
        model = POExVAE(config, obs_space, act_space)
    else:
        raise NotImplementedError()

    return model