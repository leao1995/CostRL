import torch
import torch.nn as nn
import torch.jit as jit
import numpy as np

from src.networks.embed_pool import EmbeddingPool
from src.networks.mlp import MLP
from src.models.poseq_codec import AttnLatent, RNNLatent, CatDecoder, build_prior_trans, build_posterior_trans

class POSeqVAE(nn.Module):
    def __init__(
        self,
        config,
        obs_space,
        act_space
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

        inference_input_dim = obs_embed_dim + act_embed_dim

        if config.latent_net == 'rnn':
            self.prior = RNNLatent(inference_input_dim, config.rnn_hidden_dim, config.num_rnn_layers, 
                config.prior_dims, config.latent_dim)
            self.posterior = RNNLatent(inference_input_dim, config.rnn_hidden_dim,
                config.num_rnn_layers, config.posterior_dims, config.latent_dim)
        elif config.latent_net == 'attn':
            self.prior = AttnLatent(inference_input_dim, config.attn_hidden_dims, config.prior_dims, 
                config.latent_dim)
            self.posterior = AttnLatent(inference_input_dim, config.attn_hidden_dims,
                config.posterior_dims, config.latent_dim)
        else:
            raise NotImplementedError()
        
        self.prior_trans = build_prior_trans(config.latent_dim, config.prior_trans_dims)
        
        self.posterior_trans = build_posterior_trans(config.latent_dim, config.posterior_trans_dims)

        decoder_input_dim = inference_input_dim + config.latent_dim
        self.decoder = CatDecoder(decoder_input_dim, config.decoder_dims, config.num_classes_per_dim)

    def _obs_embed(self, state, mask):
        B,T,_ = state.shape
        inputs = ((state + 1) * mask).long() # [B,T,d] 0 means unobserved
        inputs = inputs.view([B*T,-1])
        embed = self.obs_embed(inputs)
        embed = embed.view([B,T,-1])
        return embed

    def _act_prepend_and_embed(self, action):
        B,T = action.shape
        empty = torch.ones([B,1]).to(action) * -1
        action = torch.cat([empty, action], dim=1)
        embed = self.act_embed(action+1)
        return embed

    def _decode(self, inputs, latent, num_samples):
        B, T = inputs.shape[:2]
        latent = latent.view([num_samples,B,T,-1])
        inputs = inputs.unsqueeze(0).expand(num_samples,-1,-1,-1)
        decoder_inputs = torch.cat([inputs, latent], dim=-1) # [N,B,T,d]
        decoder_dist = self.decoder(decoder_inputs)
        return decoder_dist

    def forward(self, state, mask, action):
        '''
        state: [B,T,d]
        mask: [B,T,d]
        action: [B,T-1]
        '''
        B, T = state.shape[:2]
        act_embed = self._act_prepend_and_embed(action) # [B,T,d]

        # prior
        prior_obs_embed = self._obs_embed(state, mask) #[B,T,d]
        prior_inputs = torch.cat([prior_obs_embed, act_embed], dim=-1)
        prior_dist = self.prior(prior_inputs)

        # posterior
        posterior_obs_embed = self._obs_embed(state, torch.ones_like(mask))
        posterior_inputs = torch.cat([posterior_obs_embed, act_embed], dim=-1)
        posterior_dist = self.posterior(posterior_inputs)
        if self.training:
            num_posterior_samp = self.config.num_posterior_samp_train
        else:
            num_posterior_samp = self.config.num_posterior_samp_test
        posterior_samp = posterior_dist.rsample([num_posterior_samp]) #[N,B,T,d]
        log_prob = posterior_dist.log_prob(posterior_samp) # [N,B,T]
        posterior_samp = posterior_samp.view([-1]+list(posterior_samp.shape[3:]))
        posterior_samp, logabsdet = self.posterior_trans.inverse(posterior_samp)
        logabsdet = logabsdet.view([num_posterior_samp,B,T]) # [N,B,T]
        log_q_z = (log_prob - logabsdet).mean(dim=0) # [B,T]

        # kl term
        transformed_posterior_samp, logabsdet = self.prior_trans.forward(posterior_samp) #[N*B*T,d]
        transformed_posterior_samp = transformed_posterior_samp.view([num_posterior_samp,B,T,-1])
        logabsdet = logabsdet.view([num_posterior_samp,B,T]) # [N,B,T]
        log_p_z = (prior_dist.log_prob(transformed_posterior_samp) + logabsdet).mean(dim=0) #[B,T]
        kl = log_q_z - log_p_z # [B,T]

        # decoder
        decoder_inputs = prior_inputs
        decoder_dist = self._decode(decoder_inputs, posterior_samp, num_posterior_samp)
        log_likel = []
        for dist, xi in zip(decoder_dist, state.chunk(state.shape[-1], dim=-1)):
            log_likel.append(dist.log_prob(xi.squeeze(dim=-1))) #[N,B,T]
        log_likel = torch.stack(log_likel, dim=-1) # [N,B,T,d]
        log_likel = torch.sum(log_likel, dim=-1).mean(dim=0) # [B,T]
        
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

        # prior
        prior_obs_embed = self._obs_embed(state, mask) #[B,T,d]
        prior_inputs = torch.cat([prior_obs_embed, act_embed], dim=-1)
        prior_dist = self.prior(prior_inputs)
        prior_samp = prior_dist.rsample([num_samples]) # [N,B,T,d]
        prior_samp = prior_samp.view([-1]+list(prior_samp.shape[3:])) #[N*B*T,d]
        prior_samp, _ = self.prior_trans.inverse(prior_samp)

        # sampling
        decoder_inputs = prior_inputs
        decoder_dist = self._decode(decoder_inputs, prior_samp, num_samples)
        decoder_samp = [dist.logits.argmax(-1) for dist in decoder_dist]
        decoder_samp = torch.stack(decoder_samp, dim=-1).float() #[N,B,T,d]
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

