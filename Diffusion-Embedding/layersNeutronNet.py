import tensorflow as tf
from keras.layers import Layer, Dense, LayerNormalization
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from dataclasses import dataclass

class EmbedND(Layer):
    """
    Eine Schicht zur Berechnung von N-Dimensionalen Einbettungen basierend auf den Eingaben.
    
    Args:
        dim (int): Dimension der Einbettungen.
        theta (int): Skalierungsfaktor für die Frequenzen.
        axes_dim (list[int]): Dimensionen der Achsen zur Berechnung der Einbettungen.
    """
    def __init__(self, 
                 dim: int, 
                 theta: int, 
                 axes_dim: list[int]
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def call(self, inputs):
        n_axes = inputs.shape[-1]
        # Berechnet die Embedding für jede Achse
        emb = tf.concat([
            tf.signal.rfft([inputs[..., i], self.axes_dim[i], self.theta])
            for i in range(n_axes)
        ], axis=-3)

        return tf.expand_dims(emb, axis=1)

def timestep_embedding(
        t: tf.Tensor, 
        dim: int, 
        max_period: int=10000, 
        time_factor: float=1000.0
) -> tf.Tensor:
    """
    Berechnet eine Zeitembedding für eine gegebene Zeiteinbettung.

    Args:
        t (tf.Tensor): Zeitstempel.
        dim (int): Dimension der Einbettung.
        max_period (int): Maximaler Zeitraum für Frequenzberechnung.
        time_factor (float): Zeitfaktor.

    Returns:
        tf.Tensor: Zeitembeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = tf.exp(-tf.math.log(float(max_period)) * tf.range(half) / half)
    args = t[:, None] * freqs[None]
    embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
    if dim % 2 != 0:
        embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
    if tf.is_floating_point(t):
        embedding = tf.cast(embedding, t.dtype)
    return embedding

class MLPEmbedder(Layer):
    """
    Ein MLP-basiertes Einbettungs-Modul.

    Args:
        in_dim (int): Eingabedimension.
        hidden_dim (int): Dimension der versteckten Schicht.
    """
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int
    ):
        super().__init__()
        self.in_layer = Dense(hidden_dim, kernel_initializer='glorot_uniform', use_bias=True)
        self.silu = tf.nn.swish
        self.out_layer = Dense(hidden_dim, kernel_initializer='glorot_uniform', use_bias=True)

    def call(self, x):
        """
        Führt das MLP für die Eingabe `x` aus.

        Args:
            x (tf.Tensor): Eingabe-Tensor.
        
        Returns:
            tf.Tensor: Ausgabe des MLPs.
        """
        return self.out_layer(self.silu(self.in_layer(x)))

class RMSNorm(Layer):
    """
    Eine RMSNorm-Schicht zur Normalisierung.

    Args:
        dim (int): Dimension des Eingabedatensatzes.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.scale = self.add_weight(name='scale', shape=[dim])

    def call(self, x):
        """
        Normalisiert die Eingabe unter Verwendung von RMS-Norm.

        Args:
            x (tf.Tensor): Eingabe-Tensor.
        
        Returns:
            tf.Tensor: Normalisierter Tensor.
        """
        x_dtype = x.dtype
        x = tf.cast(x, 'float32')
        rrms = tf.sqrt(tf.reduce_mean(x**2, axis=-1, keepdims=True) + 1e-6)
        return (x / rrms) * self.scale

class QKNorm(Layer):
    """
    Normiert Abfrage- (Query) und Schlüssel- (Key) Tensoren.

    Args:
        dim (int): Dimension der Eingabedatensätze.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def call(self, q, k, v):
        """
        Wendet die Normalisierung auf Abfrage- und Schlüssel-Tensoren an.

        Args:
            q (tf.Tensor): Abfrage-Tensor.
            k (tf.Tensor): Schlüssel-Tensor.
            v (tf.Tensor): Wert-Tensor.

        Returns:
            tuple: Normalisierte Abfrage- und Schlüssel-Tensoren.
        """
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k

class SelfAttention(Layer):
    """
    Self-Attention Schicht, die Abfrage, Schlüssel und Wert-Tensoren kombiniert.

    Args:
        dim (int): Dimension der Eingabe.
        num_heads (int): Anzahl der Attention-Köpfe.
        qkv_bias (bool): Ob die QKV-Projektionen einen Bias verwenden sollen.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = Dense(dim * 3, use_bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = Dense(dim)

    def call(self, x, pe):
        """
        Führt die Self-Attention Berechnung aus.

        Args:
            x (tf.Tensor): Eingabe-Tensor.
            pe (tf.Tensor): Positional Embeddings.

        Returns:
            tf.Tensor: Ergebnis der Self-Attention.
        """
        qkv = self.qkv(x)
        q, k, v = tf.split(qkv, 3, axis=-1)
        q, k = self.norm(q, k, v)
        x = tf.matmul(q, k, transpose_b=True)
        x = tf.matmul(x, v)
        x = self.proj(x)
        return x

@dataclass
class ModulationOut:
    shift: tf.Tensor
    scale: tf.Tensor
    gate: tf.Tensor

class Modulation(Layer):
    """
    Eine Modulationsschicht, die Verschiebung, Skalierung und Toringoperationen durchführt.

    Args:
        dim (int): Eingabedimension.
        double (bool): Doppelte Modulation anwenden.
    """
    def __init__(self, dim, double):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = Dense(self.multiplier * dim, use_bias=True)

    def call(self, vec):
        """
        Wendet die Modulation auf die Eingabe an.

        Args:
            vec (tf.Tensor): Eingabedaten.

        Returns:
            ModulationOut: Ausgabe der Modulationsschicht.
        """
        out = self.lin(vec)[:, None, :]
        out = tf.split(out, self.multiplier, axis=-1)

        return ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None

class DoubleStreamBlock(Layer):
    """
    Ein Block für Doppelstrom-Verarbeitung, der Text- und Bilddaten kombiniert.

    Args:
        hidden_size (int): Größe der versteckten Schichten.
        num_heads (int): Anzahl der Attention-Köpfe.
        mlp_ratio (float): Verhältnis der MLP-Dimension.
        qkv_bias (bool): Ob ein Bias in der QKV-Schicht verwendet wird.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = LayerNormalization(epsilon=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = LayerNormalization(epsilon=1e-6)
        self.img_mlp = tf.keras.Sequential([
            Dense(mlp_hidden_dim, use_bias=True),
            tf.nn.swish,
            Dense(hidden_size, use_bias=True)
        ])

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = LayerNormalization(epsilon=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = LayerNormalization(epsilon=1e-6)
        self.txt_mlp = tf.keras.Sequential([
            Dense(mlp_hidden_dim, use_bias=True),
            tf.nn.swish,
            Dense(hidden_size, use_bias=True)
        ])

    def call(self, img, txt, vec, pe):
        """
        Führt die Doppelstrom-Verarbeitung auf Bild- und Textdaten aus.

        Args:
            img (tf.Tensor): Bilddaten.
            txt (tf.Tensor): Textdaten.
            vec (tf.Tensor): Modulationsvektor.
            pe (tf.Tensor): Positional Embeddings.

        Returns:
            tuple: Modifizierte Bild- und Textdaten.
        """
        # Bildmodulation und Verarbeitung
        img_mod1, img_mod2 = self.img_mod(vec)
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_attn = self.img_attn(img_modulated, pe)
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(self.img_norm2(img))

        # Textmodulation und Verarbeitung
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_attn = self.txt_attn(txt_modulated, pe)
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(self.txt_norm2(txt))

        return img, txt
