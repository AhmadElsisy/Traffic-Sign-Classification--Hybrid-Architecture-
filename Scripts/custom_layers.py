import keras
from keras import layers
from keras import activations





# Positional embedding that Keras can track
@keras.utils.register_keras_serializable(package="Custom", name="AddPositionEmbedding")
class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim   = embed_dim
        self.pos = self.add_weight(
            name="pos_embedding",
            shape=(1, num_patches, embed_dim),
            initializer="random_normal",
            trainable=True)

    def call(self, tokens):
        return tokens + self.pos

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "embed_dim":   self.embed_dim,
        })
        return {**config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
    
@keras.utils.register_keras_serializable(package="Custom", name="SwiGLU")       
class SwiGLULayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense_gate = layers.Dense(units, name="dense_gate")
        self.dense_linear = layers.Dense(units, name="dense_linear")
        
    def call(self, x):
        # Apply the first linear transformation and the sigmoid linear unit (SiLU) activation
        gate = self.dense_gate(x)
        activated_gate = activations.silu(gate)
        
        # Apply the second linear transformation
        linear = self.dense_linear(x)
        
        # Element-wise multiplication of the activated gate and the linear output
        return activated_gate * linear
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
        })
        return {**config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
# One Transformer encoder block
@keras.utils.register_keras_serializable(package="Custom", name="TransformerBlock")
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_dim, rate=0.1 ,**kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = SwiGLULayer(units=embed_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.ffn_dim = ffn_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rate = rate
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "rate": self.rate,
        })
        return {**config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
