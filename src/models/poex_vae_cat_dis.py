import torch
import torch.nn as nn
import numpy as np

from src.networks.embed_pool import EmbeddingPool
from src.models.poex_codec import *

class POExVAE(nn.Module):
    def __init__(
        self, 
        config,
        obs_space,
        act_space,
    ):
        super().__init__()
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space

        obs_high = obs_space.high
        config.num_classes_per_dim = list(map(int, obs_high + 1))
        num_obs_embeddings = list(map(int, obs_high + 2)) # extra one for unobserved
        self.obs_embed = EmbeddingPool(num_obs_embeddings, config.categorical_embed_dim)
        self.obs_embed_dim = obs_embed_dim = self.obs_embed.output_dim

        num_act_embeddings = act_space.n + 1 # extra one for empty action
        self.act_embed = nn.Embedding(num_act_embeddings, config.action_embed_dim)
        act_embed_dim = config.action_embed_dim

        inference_input_dim = obs_embed_dim + act_embed_dim + config.time_embed_dim

        self.prior = LatentEncoder(inference_input_dim, config.peq_encode_dims, config.latent_dim, 
            config.num_heads, config.num_inds, config.use_ln)
        self.prior_trans = build_prior_trans(config.latent_dim, config.prior_trans_dims)

        if config.peq_embed_dim > 0: # maybe reuse peq_embed from prior
            self.peq_embed_net = PeqEncoder(inference_input_dim, config.peq_encode_dims, config.peq_embed_dim, 
                config.num_heads, config.num_inds, config.use_ln)
        self.peq_embed_dim = peq_embed_dim = config.peq_embed_dim or self.prior.peq_embed_dim

        if config.shared_prior_posterior: # maybe use shared network
            self.posterior = self.prior
        else:
            self.posterior = LatentEncoder(inference_input_dim, config.peq_encode_dims, config.latent_dim,
                config.num_heads, config.num_inds, config.use_ln)
        self.posterior_trans = build_posterior_trans(config.latent_dim, config.posterior_trans_dims)

        # decoder may be independent across set elements, but transformer can obtain better performance
        decoder_input_dim = inference_input_dim + peq_embed_dim + config.latent_dim
        self.decoder = CatDecoder(decoder_input_dim, config.peq_decode_dims, config.num_classes_per_dim,
            config.num_heads, config.num_inds, config.use_ln)

    def _obs_embed(self, state, mask):
        B,T,_ = state.shape
        inputs = ((state + 1) * mask).long() # [B,T,d] 0 means unobserved
        embed = self.obs_embed(inputs)
        return embed

    def _act_prepend_and_embed(self, action):
        B, T = action.shape
        empty = torch.ones([B,1]).to(action) * -1
        action = torch.cat([empty, action], dim=1)
        embed = self.act_embed(action+1)
        return embed

    def _time_embed(self, B, T):
        pos = np.tile(np.arange(T), (B, 1))
        pos = np.expand_dims(pos, axis=-1)
        div_term = np.exp(np.arange(0, self.config.time_embed_dim, 2) * (-np.log(10000.0) / self.config.time_embed_dim))
        e0 = np.sin(pos * div_term)
        e1 = np.cos(pos * div_term)
        e = np.stack([e0, e1], axis=-1).reshape([pos.shape[0], pos.shape[1], self.config.time_embed_dim])

        return torch.from_numpy(e).float()

    def _decode(self, inputs, latent, num_samples):
        B, T = inputs.shape[:2]
        latent = latent.view([num_samples, -1, latent.shape[1]]) #[N,B,d]
        latent = latent.unsqueeze(2).expand(-1,-1,T,-1) #[N,B,T,d]
        decoder_inputs = inputs.unsqueeze(0).expand(num_samples,-1,-1,-1) # [N,B,T,d]
        decoder_inputs = torch.cat([decoder_inputs, latent], dim=-1)
        decoder_inputs = decoder_inputs.view([-1]+list(decoder_inputs.shape[2:])) #[N*B,T,d]
        decoder_dist = self.decoder(decoder_inputs)

        return decoder_dist

    def forward(self, state, mask, action):
        '''
        state: [B,T,d] need to be fully observed for loss computation
        mask: [B,T,d] 0: unobserved 1: observed
        action: [B,T-1] discrete actions
        '''
        B, T = state.shape[:2]
        act_embed = self._act_prepend_and_embed(action) # [B,T,d]
        time_embed = self._time_embed(B,T).to(act_embed) # [B,T,d]

        # prior
        prior_obs_embed = self._obs_embed(state, mask) #[B,T,d]
        prior_inputs = torch.cat([prior_obs_embed, act_embed, time_embed], dim=-1)
        prior_output = self.prior(prior_inputs)
        prior_dist = prior_output['dist']

        # permutation equivarient embedding
        if self.config.peq_embed_dim > 0:
            peq_embed = self.peq_embed_net(prior_inputs) #[B,T,d]
        else:
            peq_embed = prior_output['peq_embed']

        # posterior
        posterior_obs_embed = self._obs_embed(state, torch.ones_like(mask))
        posterior_inputs = torch.cat([posterior_obs_embed, act_embed, time_embed], dim=-1)
        posterior_dist = self.posterior(posterior_inputs)['dist']
        if self.training:
            num_posterior_samp = self.config.num_posterior_samp_train
        else:
            num_posterior_samp = self.config.num_posterior_samp_test
        posterior_samp = posterior_dist.rsample([num_posterior_samp]) #[N,B,...]
        log_prob = posterior_dist.log_prob(posterior_samp) # [N,B]
        posterior_samp = posterior_samp.view([-1]+list(posterior_samp.shape[2:]))
        posterior_samp, logabsdet = self.posterior_trans.inverse(posterior_samp)
        logabsdet = logabsdet.view([num_posterior_samp, -1]) # [N,B]
        log_q_z = (log_prob - logabsdet).mean(dim=0) # [B]

        # kl term
        transformed_posterior_samp, logabsdet = self.prior_trans.forward(posterior_samp) #[N*B,...]
        transformed_posterior_samp = transformed_posterior_samp.view([num_posterior_samp, -1]+list(posterior_samp.shape[1:]))
        logabsdet = logabsdet.view([num_posterior_samp, -1]) # [N,B]
        log_p_z = (prior_dist.log_prob(transformed_posterior_samp) + logabsdet).mean(dim=0) #[B]
        kl = log_q_z - log_p_z # [B]

        # decoder
        decoder_inputs = torch.cat([prior_inputs, peq_embed], dim=-1)
        decoder_dist = self._decode(decoder_inputs, posterior_samp, num_posterior_samp)
        
        # log_likelihood
        state = state.unsqueeze(0).expand(num_posterior_samp, -1, -1, -1) #[N,B,T,d]
        state = state.contiguous().view([-1]+list(state.shape[2:])) #[N*B,T,d]
        log_likel = []
        for dist, xi in zip(decoder_dist, state.chunk(state.shape[-1], dim=-1)):
            log_likel.append(dist.log_prob(xi.squeeze(dim=-1))) #[N*B,T]
        log_likel = torch.stack(log_likel, dim=-1) #[N*B,T,d]
        log_likel = log_likel.view([num_posterior_samp,-1]+list(log_likel.shape[1:])) #[N,B,T,d]
        log_likel = torch.sum(log_likel, dim=(2,3)).mean(dim=0) # [B]

        # elbo
        elbo = log_likel - kl * self.config.kl_weight

        return {
            'elbo': elbo,
            'kl': kl,
            'log_likel': log_likel
        }

    def loss(self, state, mask, action):
        results = self.forward(state, mask, action)
        
        return {
            'model_loss': -results['elbo'].mean(),
            'model_loss_kl': results['kl'].mean(),
            'model_loss_logp': results['log_likel'].mean()
        }

    @torch.no_grad()
    def impute(self, state, mask, action, num_samples):
        '''
        state: [B,T,d]
        mask: [B,T,d]
        action: [B,T-1]
        '''
        B, T = state.shape[:2]
        act_embed = self._act_prepend_and_embed(action) # [B,T,d]
        time_embed = self._time_embed(B,T).to(act_embed) # [B,T,d]

        # prior
        prior_obs_embed = self._obs_embed(state, mask) #[B,T,d]
        prior_inputs = torch.cat([prior_obs_embed, act_embed, time_embed], dim=-1)
        prior_output = self.prior(prior_inputs)
        prior_dist = prior_output['dist']
        prior_samp = prior_dist.rsample([num_samples])
        prior_samp = prior_samp.view([-1]+list(prior_samp.shape[2:])) #[N*B,d]
        prior_samp, _ = self.prior_trans.inverse(prior_samp)

        # permutation equivarient embedding
        if self.config.peq_embed_dim > 0:
            peq_embed = self.peq_embed_net(prior_inputs) #[B,T,...]
        else:
            peq_embed = prior_output['peq_embed']

        # decoder
        decoder_inputs = torch.cat([prior_inputs, peq_embed], dim=-1)
        decoder_dist = self._decode(decoder_inputs, prior_samp, num_samples)
        decoder_samp = [dist.logits.argmax(-1) for dist in decoder_dist]
        decoder_samp = torch.stack(decoder_samp, dim=-1).float()
        decoder_samp = decoder_samp.view([num_samples]+list(state.shape))
        decoder_samp = decoder_samp * (1-mask) + state * mask #[N,B,T,d]

        return decoder_samp

    @torch.no_grad()
    def accuracy(self, state, mask, action, num_samples):
        samp = self.impute(state, mask, action, num_samples)
        corr = torch.sum((samp == state) * (1-mask))
        num = torch.sum(1-mask)*num_samples + 1e-8
        acc = corr / num

        return acc

    @torch.no_grad()
    def reward(self, state, mask, action, num_samples):
        samp = self.impute(state, mask, action, num_samples)[:,:,-1] #[N,B,d]
        real = state[:,-1] #[B,d]
        m = mask[:,-1] #[B,d]
        corr = torch.sum((samp == real) * (1-m), dim=(0,2))
        num = torch.sum(1-m, dim=1)*num_samples + 1e-8
        acc = corr / num

        return acc

    @property
    def belief_dim(self):
        return self.obs_embed_dim

    @torch.no_grad()
    def belief(self, state, mask, action, num_samples, keep_last):
        decoder_samp = self.impute(state, mask, action, num_samples)
        decoder_samp = (decoder_samp + 1).long()
        decoder_samp_embed = self.obs_embed(decoder_samp)
        decoder_samp_embed = decoder_samp_embed.permute(1,2,0,3) #[B,T,N,d]
        if keep_last:
            decoder_samp_embed = decoder_samp_embed[:,-1] #[B,N,d]

        return decoder_samp_embed
    