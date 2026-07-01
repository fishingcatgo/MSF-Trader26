import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
import backtrader as bt
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym  # Import gymnasium for custom environments
from gymnasium import spaces  # Import spaces for defining observation/action spaces

from .modul.Swiglu_position import InputProcessor
from .modul.Cross_atension import MultiLayerDecoder
from .modul.Pyramid_MambaMoe import MambaMoEStackGateModel


Moe_Loss_Ratio = 0.01 # MoE loss ratio, used to control the magnitude of MoE loss

Feature_Model = "Mamba" # Feature model used: LSTM, Decoder, or Mamba
PV_UseDecoder = True # Whether to use Decoder as the PV (Policy/Value) model 
Use_PVcache = True # Whether to use PV cache to enable caching
Max_Pvcache_Que = 60 # Maximum cache queue length
LAST_LAYER_DIM_PI = 128     # Policy head output dimension
LAST_LAYER_DIM_VF = 128      # Value head output dimension


# 1. Input embedding parameters
Cfg_Input=dict(
        # input_size: int,  # Input dimension
        # hidden_size: int, # Output dimension
        max_position_embeddings=1024,  # Maximum positional embeddings
        hidden_act = "silu",  # Hidden layer activation function
        norm_eps = 1e-6, # Normalization epsilon
        
        # Topology control parameters
        use_pre_norm = True,         # Controls whether to use RMSNorm(x)
        use_pos_embedding = True,     # Controls whether to use Positional Encoding
        use_post_norm = True,        # Controls whether to use the final RMSNorm
        normalize_pos_emb = True,     # Controls whether to normalize positional embeddings
)

# 2. Mamba-MoE feature extraction parameters
Cfg_Mamba=dict(
        
        # Layers
        # hidden_size: int = features_dim, 
        num_hidden_layers = 5,  # Number of Mamba layers
        rms_norm_eps = 1e-6,  # RMSNorm epsilon

        # Mamba parameters
        d_state = 64,           # State expansion factor
        d_conv = 4,             # Local convolution width
        expand = 2,             # Block expansion factor   
        headdim = 32,           # Dimension of multi-head attention heads   

        # FFN Parameters
        use_moe = True,            # Whether to use MoE
        intermediate_size = 2,    # FFN intermediate layer multiplier relative to input
        hidden_act = "silu",         # FFN intermediate layer activation function

        # MoE Parameters
        expert_hidden_mult = 2,         # Expert hidden layer multiplier
        top_k = 2,                      # Number of experts to select
        expert_number = 4,              # Total number of experts
        shared_experts_number = 2,      # Number of shared experts

        # Downsampling parameters,Cross-attention
        use_downsampling = True, # [New] Whether to enable downsampling, layers match num_hidden_layers
        # Gate Parameters
        use_gate = True,           # [New] Whether to enable Gate module
        num_heads = 8,              # Number of Gate attention heads   
        gate_seq_len = 8,           # Only valid when query_source='token'
        num_gate_layers = 1,        # Number of layers in Gate module
        query_source = 'last_layer',     # Gate Query source: 'token' or 'last_layer' (refers to stride/step)
        gate_type = 'rope',     # Gate Attention type: 'standard' or 'rope'
        num_key_value_heads = 4,    # GQA parameters for RoPE Gate
        max_position_embeddings = 2048, # RoPE parameters for RoPE Gate
        rope_theta = 10000.0,     # RoPE parameters for RoPE Gate
)

# 3. Cross-attention feature fusion parameters
Cfg_Decoder_PV=dict(
        # hidden_size: int = 128, # Feature dimension must equal hidden_size
        num_hidden_layers = 1, # Number of hidden layers
        num_attention_heads = 8,  # Number of attention heads
        num_key_value_heads = 4,  # GQA, number of KV attention heads
        max_position_embeddings = 512,  # Maximum number of positional embeddings
        rms_norm_eps = 1e-6,  # RMSNorm epsilon
        rope_theta = 10000.0,  # RoPE theta
        attention_dropout = 0.0,  # Attention dropout
        # Switch and FFN parameters
        use_moe = False,           # Whether to use MoE
        intermediate_size = 2,      # FFN intermediate layer multiplier
        hidden_act = "silu",        # FFN intermediate activation
        # ShareMOE specific parameters
        expert_hidden_mult = 2,     # Expert hidden multiplier
        top_k = 2,                  # Experts to select
        expert_number = 4,          # Total experts
        shared_experts_number = 2,  # Shared experts   
)


# === Custom PPO Feature Extractor ===
class PPO_FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) > 1:
            self.context_len = observation_space.shape[0]
            input_dim = observation_space.shape[-1]
        else:
            self.context_len = 1 # Non-sequential data, context length set to 1
            input_dim = observation_space.shape[0]
        
        print(f"Input dim: {input_dim}, Observation shape: {observation_space.shape}, Observation dimensions: {len(observation_space.shape)}")


        Cfg_Input['input_size']=input_dim
        Cfg_Input['hidden_size']=features_dim
        print('Cfg_Input Configuration:', Cfg_Input)
        self.input_proce = InputProcessor(**Cfg_Input)

        if Feature_Model == "Mamba":
            # Pyramid structure, compatible with all Mamba architectures
            Cfg_Mamba['hidden_size']=features_dim
            print('Cfg_Mamba Configuration:', Cfg_Mamba)
            self.feature_models = MambaMoEStackGateModel(**Cfg_Mamba)       
        else:
            raise ValueError("Feature_Model must be 'LSTM' or 'Mamba' or 'Decoder'")
        
        self.feature_loss = []

    def forward(self, obs):
        self.feature_loss = []
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)

        x = self.input_proce(obs)  # [B, T, H]

        B, T, D = x.shape
        print('x dimension:', x.shape)

       
        if Feature_Model == "Mamba":
            output_moe, aux_losses_moe = self.feature_models(x)
            self.feature_loss=aux_losses_moe
      
        # Output shape [B, T, D]
        return output_moe

# Custom Policy Networks (MLP models)
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, intermediate_factor=2):
        super().__init__()
        intermediate_dim = input_dim * intermediate_factor
        self.gate_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

        self.norm = nn.RMSNorm(hidden_dim)

    def forward(self, x): # Compatible with 3D inputs
        """
        x: shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        """

        if x.dim() == 3:
            x = x[:, -1, :]  # Take the last step of the sequence

        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(gate * up)
        x = self.norm(x)
        return x  # Used directly as action output


class DPA_SwiGLUPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, intermediate_factor=2):
        super().__init__()
        intermediate_dim = input_dim * intermediate_factor
        # Path 1: SiLU gating
        self.gate_proj1 = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.up_proj1 = nn.Linear(input_dim, intermediate_dim, bias=False)
        # Path 2: GELU gating (innovation path)
        self.gate_proj2 = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.up_proj2 = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim * 2, hidden_dim, bias=False)  # Projection after fusion
        self.norm = nn.RMSNorm(hidden_dim)
        # Innovation: Simple attention fusion
        self.attn_weight = nn.Parameter(torch.ones(2))

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
            
        # Path 1 (SiLU)
        gate1 = F.silu(self.gate_proj1(x))
        up1 = self.up_proj1(x)
        path1 = gate1 * up1
        
        # Path 2 (GELU)
        gate2 = F.gelu(self.gate_proj2(x))
        up2 = self.up_proj2(x)
        path2 = gate2 * up2
        
        # Innovation: Attention-weighted fusion
        attn = F.softmax(self.attn_weight, dim=0)
        
        # Multiply weights to each path and concatenate
        # Shape transformation: [batch, 256] + [batch, 256] -> [batch, 512]
        fused = torch.cat([path1 * attn[0], path2 * attn[1]], dim=-1)

        # Weighted fusion alternative:
        # fused = attn[0] * path1 + attn[1] * path2  # [B, intermediate_dim]
        
        # Project back to hidden_dim
        x = self.down_proj(fused) 
        x = self.norm(x)
        return x

    

# === Custom PPO Policy Network ===
class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.
    :param feature_dim: dimension of the features extracted with the features_extractor
    :param last_layer_dim_pi: number of units for the last layer of the policy network
    :param last_layer_dim_vf: number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # Use global variables to ensure latest global configurations are used
        last_layer_dim_pi = LAST_LAYER_DIM_PI
        last_layer_dim_vf = LAST_LAYER_DIM_VF

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = DPA_SwiGLUPolicyNet(feature_dim, last_layer_dim_pi, intermediate_factor=2)
        
  

        if PV_UseDecoder:
            Cfg_Decoder_PV['hidden_size']=feature_dim
            print('Cfg_Decoder Configuration:', Cfg_Decoder_PV)
            self.policy_decode = MultiLayerDecoder(**Cfg_Decoder_PV)    
        
        self.value_net = SwiGLUValueNet(feature_dim, last_layer_dim_vf, intermediate_factor=2)

        if PV_UseDecoder:
            Cfg_Decoder_PV['hidden_size']=feature_dim
            print('Cfg_Decoder Configuration:', Cfg_Decoder_PV)
            self.value_decode = MultiLayerDecoder(**Cfg_Decoder_PV)

        # Save MoE losses
        self.p_loss = []
        self.v_loss = []

        # Cache settings
        self.V_Tcache = None # Value training cache
        self.P_Tcache = None # Policy training cache
        self.P_Icache = None # Policy inference cache
        self.pass_PV = False # Flag for whether data passed through PV model (for inference logic)

        self.max_queue_size = Max_Pvcache_Que # Max length of the saved queue




    # Cache reset method, called externally by Callback at the end of an episode
    def reset_Tcache(self) -> None:
        """Clear the current training feature cache."""
        print("Resetting training cache.")
        self.V_Tcache = None 
        self.P_Tcache = None 
        
    def reset_Icache(self) -> None:
        """Clear the current inference cache."""
        print("Resetting inference cache.")
        self.P_Icache = None

    def get_cache(self):
        """Retrieve current cache status."""
        return self.V_Tcache, self.P_Tcache, self.P_Icache
    
    def get_cache_shape(self):
        """Retrieve current cache dimensions."""
        return (self.V_Tcache.shape if self.V_Tcache is not None else None,
                self.P_Tcache.shape if self.P_Tcache is not None else None,
                self.P_Icache.shape if self.P_Icache is not None else None)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value.
        """
        print('Input features shape:', features.shape)
        self.pass_PV=True # Mark as passing through PV model
     
        self.p_loss = []
        self.v_loss = []
    
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor: 
        in_features = features

        if Use_PVcache:
            # Take the last step
            last_feature = features[:, -1:, :]
            if self.training:
                print("Policy - Currently in Training phase")
                if self.P_Tcache is None:
                    self.P_Tcache = last_feature
                else:
                    # Update queue, maintain fixed size
                    new_cache = torch.cat([self.P_Tcache.detach(), last_feature.detach()], dim=1)
                    if new_cache.shape[1] > self.max_queue_size:
                        self.P_Tcache = new_cache[:, -self.max_queue_size:, :]
                    else:
                        self.P_Tcache = new_cache
                
                print('P_Tcache shape:', self.P_Tcache.shape)
                in_features=self.P_Tcache

            else:
                if self.pass_PV: # If pass_PV is True, it is in sampling phase
                    print("Policy - Currently in Sampling phase")
                    in_features=last_feature
                else:
                    print("Policy - Currently in Inference phase")
                    if self.P_Icache is None:
                        self.P_Icache = last_feature
                    else:                
                         # Update queue, maintain fixed size
                        new_cache = torch.cat([self.P_Icache.detach(), last_feature.detach()], dim=1)
                        if new_cache.shape[1] > self.max_queue_size:
                            self.P_Icache = new_cache[:, -self.max_queue_size:, :]
                        else:
                            self.P_Icache = new_cache

                    in_features=self.P_Icache
                
                self.pass_PV=False # Reset pass_PV
        
            print('Policy Inference Cache:', self.P_Icache.shape if self.P_Icache is not None else None)
            print('Policy Training Cache:', self.P_Tcache.shape if self.P_Tcache is not None else None)
            print('Input Actor in_features shape:', in_features.shape)
                

        if PV_UseDecoder:
            output_moe, _, aux_losses_moe= self.policy_decode(in_features)
            self.p_loss=aux_losses_moe
            return self.policy_net(output_moe)
        
        return self.policy_net(in_features)


    def forward_critic(self, features: th.Tensor) -> th.Tensor: 
        input_features=features
        if Use_PVcache:
            # Take the last step
            last_feature = features[:, -1:, :]
            if self.training:
                print("Value - Currently in Training phase")
                if self.V_Tcache is None:
                    self.V_Tcache = last_feature
                else:                
                     # Update queue, maintain fixed size
                    new_cache = torch.cat([self.V_Tcache.detach(), last_feature.detach()], dim=1)
                    if new_cache.shape[1] > self.max_queue_size:
                        self.V_Tcache = new_cache[:, -self.max_queue_size:, :]
                    else:
                        self.V_Tcache = new_cache

                print('V_Tcache shape:', self.V_Tcache.shape)
                input_features=self.V_Tcache


            else:
                print("Value - Currently in Sampling phase")
                input_features=last_feature

            print('Value Training Cache:', self.V_Tcache.shape if self.V_Tcache is not None else None)
            print('Input Critic input_features shape:', input_features.shape)
            
        
        if PV_UseDecoder:
            output_moe, _, aux_losses_moe = self.value_decode(input_features)
            self.v_loss=aux_losses_moe
            return self.value_net(output_moe)
    
        return self.value_net(input_features)
    

class PPO_Policy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

    @property
    def pv_loss(self):
       if hasattr(self.mlp_extractor, "p_loss") and hasattr(self.mlp_extractor, "v_loss"):
           return self.mlp_extractor.p_loss, self.mlp_extractor.v_loss
       else:
           return None, None
    
    @property
    def feature_loss(self):
        if hasattr(self.features_extractor, "feature_loss"):
            return self.features_extractor.feature_loss
        else:
            return None
    
    @property
    def feature_pv_loss(self):
        if hasattr(self.features_extractor, "feature_loss") and hasattr(self.mlp_extractor, "p_loss") and hasattr(self.mlp_extractor, "v_loss"):
            return self.features_extractor.feature_loss, self.mlp_extractor.p_loss, self.mlp_extractor.v_loss
        else:
            return None, None, None




import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

class CustomPPO(PPO):
    """
    Custom PPO algorithm that adds MoE auxiliary loss to the total loss.
    """
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_loss_coef = Moe_Loss_Ratio

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())



                # MoE Loss Processing
                # Assume self.policy.feature_pv_loss returns three lists or lists of tensors
                feature_loss, p_loss, v_loss = self.policy.feature_pv_loss
                print("Policy Type:", type(self.policy))
                print("Captured in train:", "feature_loss:", feature_loss, "p_loss:", p_loss, "v_loss:", v_loss)
                print(type(policy_loss), policy_loss)

                # --- Calculate the mean of all three, skipping None or empty lists ---
                valid_lists = [lst for lst in [feature_loss, p_loss, v_loss] if lst not in [None, []]]
                all_losses = [l for lst in valid_lists for l in lst]

                if all_losses:
                    all_loss = torch.stack(all_losses).mean()
                else:
                    all_loss = torch.tensor(0.0, device=policy_loss.device)  # Avoid empty list error

                # --- Total Loss ---
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.moe_loss_coef * all_loss  # Includes averaged feature/p/v loss
                )

                print("MOE Loss:", all_loss)
                print("Scaled MOE Loss:", self.moe_loss_coef * all_loss)
                print("Total Loss:", loss)
                            
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)




from stable_baselines3.common.callbacks import BaseCallback

# Define callback class to reset cache at the end of an episode in a single environment
class Cache_ResetCallback(BaseCallback):
    """
    A custom callback that resets the feature cache in CustomNetwork
    at the end of each episode.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method is called by the algorithm after every step in the environment.
        """

        done = self.locals["dones"][0]
        print(f"Done Status: {type(done)}, {done}")
        # If the episode is finished
        if done:
            # Confirm policy network is our custom network and has reset_Tcache method
            if hasattr(self.model.policy.mlp_extractor, 'reset_Tcache'):
                # Call reset_Tcache method
                self.model.policy.mlp_extractor.reset_Tcache()
                print("Training episode done, training feature cache reset.")
                V_T, P_T, P_I = self.model.policy.mlp_extractor.get_cache_shape()
                print(f"Post-reset cache shapes: V_T={V_T}, P_T={P_T}, P_I={P_I}")
        return True
    


if __name__ == "__main__":
    # Replace the standard PPO with the CustomPPO class
    PPO=CustomPPO

    policy_kwargs = dict(
                features_extractor_class=PPO_FeatureExtractor,
                features_extractor_kwargs=dict(features_dim=512),
            )
     # === 1. Test model construction and training ===
    model = PPO(
        PPO_Policy, 
        "CartPole-v1", 
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=6,    # Number of sampling steps
        batch_size=3, # Number of samples processed per batch
        n_epochs=2,   # Number of updates per sample batch
        )
    callback=Cache_ResetCallback()
    model.learn(10, progress_bar=True, callback=callback)

    print("*" * 50)
    print('Model Inference:') # Inference batch size is 1
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(10):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        
        if done:
            if hasattr(model.policy.mlp_extractor, 'reset_Icache'):
                    # Call reset_Icache method
                    model.policy.mlp_extractor.reset_Icache()
                    print("Inference episode done, feature cache reset.")
    print("*" * 50)

    print("Training complete. Model policy structure:")
    print(model.policy)

    # === Model Summary (Optional debugging step) ===
    print("Model Summary:")
    try:
        from torchinfo import summary
        vec_env = model.get_env()
        obs = vec_env.reset()
        summary(model.policy, input_size=obs.shape)
    except ImportError:
        print("Please run `pip install torchinfo` to see the model summary.")