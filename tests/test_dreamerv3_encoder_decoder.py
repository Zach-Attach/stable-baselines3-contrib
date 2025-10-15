"""
Unit tests for DreamerV3 Encoder and Decoder.
"""

import gymnasium as gym
import pytest
import torch as th

from sb3_contrib.common.dreamerv3.encoder_decoder import Decoder, Encoder


class TestEncoder:
    """Test Encoder network."""

    @pytest.fixture
    def obs_space_vector(self):
        """Vector observation space."""
        return gym.spaces.Box(low=-1, high=1, shape=(4,))

    @pytest.fixture
    def obs_space_image(self):
        """Image observation space."""
        return gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype="uint8")

    @pytest.fixture
    def encoder_vector(self, obs_space_vector):
        """Encoder for vector observations."""
        return Encoder(
            observation_space=obs_space_vector,
            hidden_dim=128,
            num_layers=2,
        )

    @pytest.fixture
    def encoder_image(self, obs_space_image):
        """Encoder for image observations."""
        return Encoder(
            observation_space=obs_space_image,
            hidden_dim=256,
            num_layers=3,
        )

    def test_vector_encoder_initialization(self, encoder_vector):
        """Test vector encoder initialization."""
        assert encoder_vector.obs_type == "vector"
        assert encoder_vector.hidden_dim == 128
        assert encoder_vector.num_layers == 2

    def test_image_encoder_initialization(self, encoder_image):
        """Test image encoder initialization."""
        assert encoder_image.obs_type == "image"
        assert encoder_image.hidden_dim == 256

    def test_vector_encoder_forward(self, encoder_vector):
        """Test vector encoder forward pass."""
        batch_size = 4
        obs = th.randn(batch_size, 4)
        
        tokens = encoder_vector(obs)
        
        assert tokens.shape == (batch_size, encoder_vector.hidden_dim)
        assert not th.isnan(tokens).any()

    def test_image_encoder_forward(self, encoder_image):
        """Test image encoder forward pass."""
        batch_size = 4
        obs = th.randint(0, 255, (batch_size, 3, 64, 64), dtype=th.uint8)
        
        tokens = encoder_image(obs)
        
        assert tokens.shape == (batch_size, encoder_image.hidden_dim)
        assert not th.isnan(tokens).any()

    def test_symlog_transformation(self):
        """Test symlog transformation."""
        x = th.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        
        transformed = Encoder.symlog(x)
        
        # Symlog should preserve sign
        assert th.all((x >= 0) == (transformed >= 0))
        # Symlog of 0 should be 0
        assert transformed[2] == 0
        # Symlog should compress large values
        assert th.abs(transformed[0]) < th.abs(x[0])
        assert th.abs(transformed[4]) < th.abs(x[4])


class TestDecoder:
    """Test Decoder network."""

    @pytest.fixture
    def obs_space_vector(self):
        """Vector observation space."""
        return gym.spaces.Box(low=-1, high=1, shape=(4,))

    @pytest.fixture
    def obs_space_image(self):
        """Image observation space."""
        return gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype="uint8")

    @pytest.fixture
    def decoder_vector(self, obs_space_vector):
        """Decoder for vector observations."""
        feature_dim = 512
        return Decoder(
            observation_space=obs_space_vector,
            feature_dim=feature_dim,
            hidden_dim=128,
            num_layers=2,
        )

    @pytest.fixture
    def decoder_image(self, obs_space_image):
        """Decoder for image observations."""
        feature_dim = 512
        return Decoder(
            observation_space=obs_space_image,
            feature_dim=feature_dim,
            hidden_dim=256,
            num_layers=3,
        )

    def test_vector_decoder_initialization(self, decoder_vector):
        """Test vector decoder initialization."""
        assert decoder_vector.obs_type == "vector"
        assert decoder_vector.hidden_dim == 128
        assert decoder_vector.num_layers == 2

    def test_image_decoder_initialization(self, decoder_image):
        """Test image decoder initialization."""
        assert decoder_image.obs_type == "image"
        assert decoder_image.hidden_dim == 256

    def test_vector_decoder_forward(self, decoder_vector):
        """Test vector decoder forward pass."""
        batch_size = 4
        features = th.randn(batch_size, decoder_vector.feature_dim)
        
        recon = decoder_vector(features)
        
        assert recon.shape == (batch_size, 4)
        assert not th.isnan(recon).any()

    def test_image_decoder_forward(self, decoder_image):
        """Test image decoder forward pass."""
        batch_size = 4
        features = th.randn(batch_size, decoder_image.feature_dim)
        
        recon = decoder_image(features)
        
        # Output shape should match observation space
        assert recon.shape == (batch_size, 3, 64, 64)
        assert not th.isnan(recon).any()

    def test_vector_reconstruction_loss(self, decoder_vector):
        """Test reconstruction loss for vectors."""
        batch_size = 4
        recon = th.randn(batch_size, 4)
        target = th.randn(batch_size, 4)
        
        loss = decoder_vector.reconstruction_loss(recon, target)
        
        assert isinstance(loss, th.Tensor)
        assert loss.shape == ()  # Scalar
        assert loss >= 0

    def test_image_reconstruction_loss(self, decoder_image):
        """Test reconstruction loss for images."""
        batch_size = 4
        recon = th.randn(batch_size, 3, 64, 64)  # Logits
        target = th.randint(0, 255, (batch_size, 3, 64, 64), dtype=th.uint8)
        
        loss = decoder_image.reconstruction_loss(recon, target)
        
        assert isinstance(loss, th.Tensor)
        assert loss.shape == ()
        assert loss >= 0


class TestEncoderDecoderIntegration:
    """Test Encoder-Decoder integration."""

    def test_vector_encode_decode_cycle(self):
        """Test encoding and decoding vector observations."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        hidden_dim = 128
        
        encoder = Encoder(
            observation_space=obs_space,
            hidden_dim=hidden_dim,
        )
        
        decoder = Decoder(
            observation_space=obs_space,
            feature_dim=hidden_dim,
            hidden_dim=64,
        )
        
        batch_size = 4
        obs = th.randn(batch_size, 4)
        
        # Encode
        tokens = encoder(obs)
        assert tokens.shape == (batch_size, hidden_dim)
        
        # Decode
        recon = decoder(tokens)
        assert recon.shape == (batch_size, 4)
        
        # Compute loss
        loss = decoder.reconstruction_loss(recon, obs)
        assert loss >= 0

    def test_image_encode_decode_cycle(self):
        """Test encoding and decoding image observations."""
        obs_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype="uint8")
        hidden_dim = 256
        
        encoder = Encoder(
            observation_space=obs_space,
            hidden_dim=hidden_dim,
        )
        
        decoder = Decoder(
            observation_space=obs_space,
            feature_dim=hidden_dim,
            hidden_dim=128,
        )
        
        batch_size = 4
        obs = th.randint(0, 255, (batch_size, 3, 64, 64), dtype=th.uint8)
        
        # Encode
        tokens = encoder(obs)
        assert tokens.shape == (batch_size, hidden_dim)
        
        # Decode
        recon = decoder(tokens)
        assert recon.shape == (batch_size, 3, 64, 64)
        
        # Compute loss
        loss = decoder.reconstruction_loss(recon, obs)
        assert loss >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
