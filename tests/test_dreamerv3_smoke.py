"""
Smoke test for DreamerV3 - quick verification that basic training works.
"""

import gymnasium as gym
import pytest

from sb3_contrib import DreamerV3


def test_dreamerv3_smoke_test_cartpole():
    """
    Smoke test: Train DreamerV3 on CartPole for a few steps.
    
    This test verifies that:
    1. Model can be instantiated
    2. Components are set up correctly
    3. Training loop executes without errors
    4. Model can make predictions
    """
    env = gym.make("CartPole-v1")
    
    # Create model with small parameters for fast testing
    model = DreamerV3(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        batch_size=4,
        batch_length=10,
        imagination_horizon=5,
        learning_starts=50,
        buffer_size=1000,
        verbose=0,
    )
    
    # Train for a short time
    model.learn(total_timesteps=100)
    
    # Test prediction
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    
    assert action is not None
    
    env.close()
    print("✅ Smoke test passed: DreamerV3 trains and predicts successfully")


def test_dreamerv3_components_smoke():
    """Smoke test for individual components."""
    import torch as th
    
    env = gym.make("CartPole-v1")
    model = DreamerV3("MlpPolicy", env, learning_starts=10, verbose=0)
    model._setup_dreamerv3_components()
    
    # Test RSSM
    batch_size = 2
    carry = model.rssm.initial(batch_size, model.device)
    assert carry is not None
    
    # Test encoder
    obs = th.randn(batch_size, 4)
    tokens = model.encoder(obs)
    assert tokens.shape[0] == batch_size
    
    # Test decoder
    deter = th.randn(batch_size, model.rssm.deter_dim)
    stoch = th.randn(batch_size, model.rssm.stoch_dim, model.rssm.num_classes)
    feat = model.rssm.get_feat(deter, stoch)
    recon = model.decoder(feat)
    assert recon.shape == obs.shape
    
    env.close()
    print("✅ Component smoke test passed: All components work independently")


if __name__ == "__main__":
    # Can run directly for quick testing
    test_dreamerv3_smoke_test_cartpole()
    test_dreamerv3_components_smoke()
    print("\n" + "="*60)
    print("All smoke tests passed!")
    print("="*60)
