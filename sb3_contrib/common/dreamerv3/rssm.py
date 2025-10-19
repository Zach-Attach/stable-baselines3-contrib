"""
RSSM (Recurrent State-Space Model) for DreamerV3.

This module implements the world model component of DreamerV3, consisting of:
- Deterministic path: Block-diagonal GRU (4096 dim, 8 blocks)
- Stochastic path: 32 categorical distributions with 32 classes each
- Prior network: p(s_t | h_t)
- Posterior network: q(s_t | h_t, o_t)
"""

from typing import Dict, Optional, Tuple

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from sb3_contrib.common.dreamerv3.policies import BlockDiagonalGRU


class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM) - the world model core of DreamerV3.
    
    The RSSM consists of:
    - Deterministic state (deter): Updated by a block-diagonal GRU
    - Stochastic state (stoch): Sampled from categorical distributions
    - Prior network: Predicts stochastic state from deterministic state
    - Posterior network: Predicts stochastic state from deterministic state and observations
    
    :param action_space: Action space of the environment
    :param deter_dim: Dimension of deterministic state (default: 4096)
    :param stoch_dim: Number of stochastic variables (default: 32)
    :param num_classes: Number of classes per stochastic variable (default: 32)
    :param hidden_dim: Hidden layer dimension (default: 2048)
    :param num_blocks: Number of blocks in block-diagonal GRU (default: 8)
    :param activation: Activation function (default: nn.SiLU for DreamerV3)
    :param unimix: Uniform mixing coefficient for categorical distributions (default: 0.01)
    :param free_nats: Free nats for KL divergence (minimum KL) (default: 1.0)
    :param num_prior_layers: Number of layers in prior network (default: 2)
    :param num_posterior_layers: Number of layers in posterior network (default: 1)
    """
    
    def __init__(
        self,
        action_space: spaces.Space,
        deter_dim: int = 4096,
        stoch_dim: int = 32,
        num_classes: int = 32,
        hidden_dim: int = 2048,
        num_blocks: int = 8,
        activation: nn.Module = nn.SiLU,
        unimix: float = 0.01,
        free_nats: float = 1.0,
        num_prior_layers: int = 2,
        num_posterior_layers: int = 1,
    ):
        super().__init__()
        
        self.action_space = action_space
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.activation = activation
        self.unimix = unimix
        self.free_nats = free_nats
        self.num_prior_layers = num_prior_layers
        self.num_posterior_layers = num_posterior_layers
        
        # Get action dimension
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
            self.discrete_action = True
        elif isinstance(action_space, spaces.Box):
            self.action_dim = action_space.shape[0]
            self.discrete_action = False
        else:
            raise NotImplementedError(f"Action space {action_space} not supported")
        
        # Total stochastic state size (flattened)
        self.stoch_size = stoch_dim * num_classes
        
        # Build deterministic path (GRU core)
        # Input to GRU: processed deter + stoch + action
        self._build_gru_core()
        
        # Build posterior network (observation encoder)
        self._build_posterior()
        
        # Build prior network
        self._build_prior()
        
    def _build_gru_core(self):
        """Build the block-diagonal GRU core for deterministic state updates."""
        # Input processing for GRU
        # Process previous deter, stoch, and action separately then concatenate
        self.deter_proj = nn.Sequential(
            nn.Linear(self.deter_dim, self.hidden_dim),
            nn.RMSNorm(self.hidden_dim),
            self.activation(),
        )
        
        self.stoch_proj = nn.Sequential(
            nn.Linear(self.stoch_size, self.hidden_dim),
            nn.RMSNorm(self.hidden_dim),
            self.activation(),
        )
        
        self.action_proj = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.RMSNorm(self.hidden_dim),
            self.activation(),
        )
        
        # Block-diagonal GRU
        # Input size: 3 * hidden_dim (concatenated projections)
        # Hidden size: deter_dim
        gru_input_dim = 3 * self.hidden_dim
        self.gru = BlockDiagonalGRU(gru_input_dim, self.deter_dim, self.num_blocks)
        
    def _build_posterior(self):
        """Build posterior network q(s_t | h_t, o_t)."""
        # Input: concatenation of deter and tokens (observations)
        # We'll determine token_dim dynamically during forward pass
        # For now, create layers that will be used
        self.posterior_layers = nn.ModuleList()
        for i in range(self.num_posterior_layers):
            # These will be created dynamically on first forward pass
            self.posterior_layers.append(None)
        self.posterior_logits = None  # Will be created dynamically
        
    def _build_prior(self):
        """Build prior network p(s_t | h_t)."""
        layers = []
        input_dim = self.deter_dim
        
        for i in range(self.num_prior_layers):
            layers.extend([
                nn.Linear(input_dim, self.hidden_dim),
                nn.RMSNorm(self.hidden_dim),
                self.activation(),
            ])
            input_dim = self.hidden_dim
        
        self.prior_net = nn.Sequential(*layers)
        self.prior_logits = nn.Linear(self.hidden_dim, self.stoch_dim * self.num_classes)
        
    def _create_posterior_layer(self, input_dim: int, layer_idx: int):
        """Dynamically create posterior layer."""
        layer = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.RMSNorm(self.hidden_dim),
            self.activation(),
        )
        return layer
        
    def entry_space(self) -> Dict[str, Tuple]:
        """
        Return the space specification for RSSM entries (for replay context).
        
        :return: Dict with 'deter' and 'stoch' shapes and dtypes
        """
        return {
            'deter': ((self.deter_dim,), th.float32),
            'stoch': ((self.stoch_dim, self.num_classes), th.float32),
        }
    
    def initial(self, batch_size: int, device: th.device) -> Dict[str, th.Tensor]:
        """
        Initialize RSSM state.
        
        :param batch_size: Batch size
        :param device: Device to create tensors on
        :return: Initial state dict with 'deter' and 'stoch'
        """
        deter = th.zeros(batch_size, self.deter_dim, device=device)
        stoch = th.zeros(batch_size, self.stoch_dim, self.num_classes, device=device)
        return {'deter': deter, 'stoch': stoch}
    
    def truncate(self, entries: Dict[str, th.Tensor], carry: Optional[Dict[str, th.Tensor]] = None) -> Dict[str, th.Tensor]:
        """
        Truncate entries to get carry state for replay context.
        
        Takes the last timestep from entries sequence.
        
        :param entries: Dict with 'deter' and 'stoch' of shape (batch, time, ...)
        :param carry: Optional current carry (unused, for API compatibility)
        :return: Carry state dict with shape (batch, ...)
        """
        assert entries['deter'].ndim == 3, f"Expected 3D tensor, got shape {entries['deter'].shape}"
        # Extract last timestep
        return {k: v[:, -1] for k, v in entries.items()}
    
    def observe(
        self,
        carry: Dict[str, th.Tensor],
        tokens: th.Tensor,
        actions: th.Tensor,
        resets: th.Tensor,
        training: bool = True,
    ) -> Tuple[Dict[str, th.Tensor], Dict[str, th.Tensor], Dict[str, th.Tensor]]:
        """
        Process observations through RSSM (posterior path).
        
        :param carry: Previous state dict {'deter': (B, deter_dim), 'stoch': (B, stoch_dim, num_classes)}
        :param tokens: Encoded observations (B, T, token_dim)
        :param actions: Actions taken (B, T, action_dim)
        :param resets: Episode reset flags (B, T)
        :param training: Whether in training mode
        :return: Tuple of (final_carry, entries, features)
        """
        B, T, _ = tokens.shape
        device = tokens.device
        
        # Initialize lists to store sequence
        deter_seq = []
        stoch_seq = []
        logits_seq = []
        
        # Current state
        deter = carry['deter']
        stoch = carry['stoch']
        
        # Process sequence
        for t in range(T):
            # Get current timestep data
            token_t = tokens[:, t]  # (B, token_dim)
            action_t = actions[:, t]  # (B, action_dim) or (B,) for discrete
            
            # Ensure token_t is 2D (defensive check for B=1 case)
            if token_t.ndim == 1:
                token_t = token_t.unsqueeze(0)
            
            # Ensure action_t is 2D for consistent handling
            if action_t.ndim == 1:
                action_t = action_t.unsqueeze(-1)  # (B,) -> (B, 1)
            
            reset_t = resets[:, t].unsqueeze(-1).float()  # (B, 1), convert to float for arithmetic
            
            # Reset state if episode boundary
            # Keep the shape explicitly to avoid squeeze issues when B=1
            # Use keepdim operations to maintain dimensions
            mask = (1.0 - reset_t)  # (B, 1)
            deter = deter * mask  # (B, deter_dim) * (B, 1) -> (B, deter_dim)
            stoch = stoch * mask.unsqueeze(-1)  # (B, stoch_dim, num_classes) * (B, 1, 1)
            action_t = action_t * mask  # (B, action_dim) * (B, 1)
            
            # Ensure deter maintains 2D shape (defensive check)
            if deter.ndim == 1:
                deter = deter.unsqueeze(0)
            
            # Update deterministic state
            deter = self._core(deter, stoch, action_t)
            
            # CRITICAL: Ensure deter is 2D before calling _posterior
            # This handles edge cases where operations might reduce dimensions
            if deter.ndim == 1:
                deter = deter.unsqueeze(0)  # (deter_dim,) -> (1, deter_dim)
            assert deter.ndim == 2, f"deter must be 2D, got shape {deter.shape}"
            
            # Ensure token_t is also 2D
            if token_t.ndim == 1:
                token_t = token_t.unsqueeze(0)  # (token_dim,) -> (1, token_dim)
            assert token_t.ndim == 2, f"token_t must be 2D, got shape {token_t.shape}"
            
            # Compute posterior distribution
            logits = self._posterior(deter, token_t)
            
            # Sample stochastic state
            if training:
                stoch = self._sample_categorical(logits)
            else:
                # Use mode during inference
                stoch = self._mode_categorical(logits)
            
            # Store
            deter_seq.append(deter)
            stoch_seq.append(stoch)
            logits_seq.append(logits)
        
        # Stack sequences
        deter_seq = th.stack(deter_seq, dim=1)  # (B, T, deter_dim)
        stoch_seq = th.stack(stoch_seq, dim=1)  # (B, T, stoch_dim, num_classes)
        logits_seq = th.stack(logits_seq, dim=1)  # (B, T, stoch_dim, num_classes)
        
        # Final carry
        final_carry = {'deter': deter, 'stoch': stoch}
        
        # Entries (for storing in replay buffer)
        entries = {'deter': deter_seq, 'stoch': stoch_seq}
        
        # Features (including logits for loss computation)
        features = {
            'deter': deter_seq,
            'stoch': stoch_seq,
            'logits': logits_seq,
        }
        
        return final_carry, entries, features
    
    def imagine(
        self,
        carry: Dict[str, th.Tensor],
        policy: nn.Module,
        horizon: int,
        training: bool = True,
    ) -> Tuple[Dict[str, th.Tensor], Dict[str, th.Tensor], th.Tensor]:
        """
        Imagine future trajectories using the policy (prior path).
        
        :param carry: Starting state dict
        :param policy: Policy network to generate actions
        :param horizon: Number of steps to imagine
        :param training: Whether in training mode
        :return: Tuple of (final_carry, features, actions)
        """
        # Initialize lists
        deter_seq = []
        stoch_seq = []
        logits_seq = []
        action_seq = []
        
        # Current state
        deter = carry['deter']
        stoch = carry['stoch']
        
        # Imagine sequence
        for h in range(horizon):
            # Get action from policy
            # Policy takes feature (deter + stoch concatenated)
            with th.no_grad() if not training else th.enable_grad():
                feat = self.get_feat(deter, stoch)
                action = policy(feat)
            
            # Update deterministic state
            deter = self._core(deter, stoch, action)
            
            # Compute prior distribution
            logits = self._prior(deter)
            
            # Sample stochastic state
            if training:
                stoch = self._sample_categorical(logits)
            else:
                stoch = self._mode_categorical(logits)
            
            # Store
            deter_seq.append(deter)
            stoch_seq.append(stoch)
            logits_seq.append(logits)
            action_seq.append(action)
        
        # Stack sequences
        deter_seq = th.stack(deter_seq, dim=1)
        stoch_seq = th.stack(stoch_seq, dim=1)
        logits_seq = th.stack(logits_seq, dim=1)
        action_seq = th.stack(action_seq, dim=1)
        
        # Final carry
        final_carry = {'deter': deter, 'stoch': stoch}
        
        # Features
        features = {
            'deter': deter_seq,
            'stoch': stoch_seq,
            'logits': logits_seq,
        }
        
        return final_carry, features, action_seq
    
    def _core(self, deter: th.Tensor, stoch: th.Tensor, action: th.Tensor) -> th.Tensor:
        """
        GRU core: Update deterministic state.
        
        :param deter: Current deterministic state (B, deter_dim)
        :param stoch: Current stochastic state (B, stoch_dim, num_classes)
        :param action: Action (B, action_dim) or (B, 1) for discrete
        :return: New deterministic state (B, deter_dim)
        """
        # Ensure inputs are 2D+ with batch dimension
        assert deter.ndim >= 2, f"deter must be at least 2D, got shape {deter.shape}"
        assert stoch.ndim >= 2, f"stoch must be at least 2D, got shape {stoch.shape}"
        assert action.ndim >= 2, f"action must be at least 2D, got shape {action.shape}"
        
        # Flatten stochastic state
        stoch_flat = stoch.reshape(stoch.shape[0], -1)
        
        # Handle discrete actions - convert to one-hot if needed
        if self.discrete_action and action.shape[-1] == 1:
            # Action is an index, convert to one-hot
            action = action.long().squeeze(-1)
            action = F.one_hot(action, num_classes=self.action_dim).float()
        
        # Normalize action (as in original)
        action = action / th.maximum(th.ones_like(action), th.abs(action))
        
        # Project inputs
        deter_proj = self.deter_proj(deter)
        stoch_proj = self.stoch_proj(stoch_flat)
        action_proj = self.action_proj(action)
        
        # Concatenate
        gru_input = th.cat([deter_proj, stoch_proj, action_proj], dim=-1)
        
        # GRU update (single step)
        gru_input_seq = gru_input.unsqueeze(0)  # (1, B, input_dim)
        # Don't pass h_0, let GRU use deter as the previous hidden state
        # Actually, we need to reconstruct the hidden state from current deter
        # The GRU expects (seq_len, batch, input) and returns (seq_len, batch, hidden)
        output_seq, new_deter_h = self.gru(gru_input_seq, deter.unsqueeze(0))
        
        # Remove sequence dimension but keep batch dimension
        # new_deter_h has shape (1, B, deter_dim) or (num_layers, B, deter_dim)
        # We want (B, deter_dim)
        if new_deter_h.ndim == 3:
            result = new_deter_h[0]  # (B, deter_dim)
        elif new_deter_h.ndim == 2:
            result = new_deter_h  # Already (B, deter_dim)
        else:
            raise ValueError(f"Unexpected GRU output shape: {new_deter_h.shape}")
        
        # If still wrong (B=1 edge case), force it to be 2D
        if result.ndim == 1:
            result = result.unsqueeze(0)  # (deter_dim,) -> (1, deter_dim)
        
        # Ensure result is 2D
        assert result.ndim == 2, f"_core must return 2D tensor, got shape {result.shape}"
        return result
    
    def _posterior(self, deter: th.Tensor, tokens: th.Tensor) -> th.Tensor:
        """
        Posterior network: q(s_t | h_t, o_t).
        
        :param deter: Deterministic state (B, deter_dim)
        :param tokens: Observation tokens (B, token_dim)
        :return: Logits for categorical distributions (B, stoch_dim, num_classes)
        """
        # Concatenate deter and tokens
        x = th.cat([deter, tokens], dim=-1)
        
        # Dynamically create layers if needed
        if self.posterior_layers[0] is None:
            input_dim = x.shape[-1]
            for i in range(self.num_posterior_layers):
                self.posterior_layers[i] = self._create_posterior_layer(
                    input_dim if i == 0 else self.hidden_dim, i
                ).to(x.device)
            self.posterior_logits = nn.Linear(
                self.hidden_dim, self.stoch_dim * self.num_classes
            ).to(x.device)
        
        # Forward through layers
        for layer in self.posterior_layers:
            x = layer(x)
        
        # Get logits
        logits = self.posterior_logits(x)
        logits = logits.reshape(-1, self.stoch_dim, self.num_classes)
        
        return logits
    
    def _prior(self, deter: th.Tensor) -> th.Tensor:
        """
        Prior network: p(s_t | h_t).
        
        :param deter: Deterministic state (B, deter_dim)
        :return: Logits for categorical distributions (B, stoch_dim, num_classes)
        """
        x = self.prior_net(deter)
        logits = self.prior_logits(x)
        logits = logits.reshape(-1, self.stoch_dim, self.num_classes)
        return logits
    
    def _sample_categorical(self, logits: th.Tensor) -> th.Tensor:
        """
        Sample from categorical distributions with unimix.
        
        :param logits: Logits (B, stoch_dim, num_classes)
        :return: One-hot samples (B, stoch_dim, num_classes)
        """
        # Apply unimix (uniform mixing)
        if self.unimix > 0:
            probs = F.softmax(logits, dim=-1)
            uniform = th.ones_like(probs) / self.num_classes
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = th.log(probs + 1e-8)
        
        # Sample
        dist = th.distributions.Categorical(logits=logits)
        samples = dist.sample()
        
        # Convert to one-hot
        one_hot = F.one_hot(samples, num_classes=self.num_classes).float()
        
        return one_hot
    
    def _mode_categorical(self, logits: th.Tensor) -> th.Tensor:
        """
        Get mode of categorical distributions.
        
        :param logits: Logits (B, stoch_dim, num_classes)
        :return: One-hot mode (B, stoch_dim, num_classes)
        """
        # Get argmax
        mode = th.argmax(logits, dim=-1)
        
        # Convert to one-hot
        one_hot = F.one_hot(mode, num_classes=self.num_classes).float()
        
        return one_hot
    
    def kl_loss(
        self,
        posterior_logits: th.Tensor,
        prior_logits: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute KL divergence losses for representation learning.
        
        :param posterior_logits: Posterior logits (B, T, stoch_dim, num_classes)
        :param prior_logits: Prior logits (B, T, stoch_dim, num_classes)
        :return: Tuple of (dynamics_loss, representation_loss)
        """
        # Flatten batch and time dimensions for easier computation
        B, T, S, C = posterior_logits.shape
        post_flat = posterior_logits.reshape(B * T, S, C)
        prior_flat = prior_logits.reshape(B * T, S, C)
        
        # Dynamics loss: KL(posterior || prior) - encourages prior to match posterior
        # Stop gradient on posterior
        posterior_dist = th.distributions.Categorical(logits=post_flat.detach())
        prior_dist = th.distributions.Categorical(logits=prior_flat)
        dyn_kl = th.distributions.kl_divergence(posterior_dist, prior_dist)
        
        # Representation loss: KL(prior || posterior) - encourages posterior to match prior
        # Stop gradient on prior
        posterior_dist_2 = th.distributions.Categorical(logits=post_flat)
        prior_dist_2 = th.distributions.Categorical(logits=prior_flat.detach())
        rep_kl = th.distributions.kl_divergence(prior_dist_2, posterior_dist_2)
        
        # Apply free nats (minimum KL)
        if self.free_nats > 0:
            dyn_kl = th.maximum(dyn_kl, th.tensor(self.free_nats, device=dyn_kl.device))
            rep_kl = th.maximum(rep_kl, th.tensor(self.free_nats, device=rep_kl.device))
        
        # Sum over stochastic dimensions, mean over batch and time
        dyn_loss = dyn_kl.sum(dim=-1).mean()
        rep_loss = rep_kl.sum(dim=-1).mean()
        
        return dyn_loss, rep_loss
    
    def get_feat(self, deter: th.Tensor, stoch: th.Tensor) -> th.Tensor:
        """
        Concatenate deterministic and stochastic states to form feature vector.
        
        :param deter: Deterministic state (B, deter_dim) or (B, T, deter_dim)
        :param stoch: Stochastic state (B, stoch_dim, num_classes) or (B, T, stoch_dim, num_classes)
        :return: Feature vector (B, deter_dim + stoch_size) or (B, T, deter_dim + stoch_size)
        """
        # Handle both 2D and 3D inputs (with or without time dimension)
        if deter.ndim == 3 and stoch.ndim == 4:
            # Has time dimension: (B, T, ...)
            B, T = deter.shape[:2]
            # Verify dimensions
            if deter.shape[2] != self.deter_dim:
                raise ValueError(f"deter last dim should be {self.deter_dim}, got {deter.shape}")
            if stoch.shape[2] != self.stoch_dim or stoch.shape[3] != self.num_classes:
                raise ValueError(f"stoch dims should be [{self.stoch_dim}, {self.num_classes}], got {stoch.shape}")
            stoch_flat = stoch.reshape(B, T, -1)
        elif deter.ndim == 2 and stoch.ndim == 3:
            # No time dimension: (B, ...)
            if deter.shape[1] != self.deter_dim:
                raise ValueError(f"deter last dim should be {self.deter_dim}, got {deter.shape}")
            if stoch.shape[1] != self.stoch_dim or stoch.shape[2] != self.num_classes:
                raise ValueError(f"stoch dims should be [{self.stoch_dim}, {self.num_classes}], got {stoch.shape}")
            stoch_flat = stoch.reshape(stoch.shape[0], -1)
        else:
            raise ValueError(f"Incompatible shapes: deter {deter.shape}, stoch {stoch.shape}")
        
        return th.cat([deter, stoch_flat], dim=-1)
    
    def get_dist(self, logits: th.Tensor) -> th.distributions.Categorical:
        """
        Get categorical distribution from logits.
        
        :param logits: Logits (B, stoch_dim, num_classes)
        :return: Categorical distribution
        """
        return th.distributions.Categorical(logits=logits)
