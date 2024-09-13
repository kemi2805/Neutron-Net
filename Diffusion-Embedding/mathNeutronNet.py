import tensorflow as tf
import einops

def attention(
        q: tf.Tensor,
        k: tf.Tensor, 
        v: tf.Tensor, 
        pe: tf.Tensor
) -> tf.Tensor:
    """
    Berechnet die Attention unter Verwendung von Abfrage (q), Schlüssel (k), Wert (v) und Positional Encoding (pe).
    
    Args:
        q (tf.Tensor): Abfrage-Tensor der Form (B, H, L, D).
        k (tf.Tensor): Schlüssel-Tensor der Form (B, H, L, D).
        v (tf.Tensor): Wert-Tensor der Form (B, H, L, D).
        pe (tf.Tensor): Positional Encoding-Tensor der Form (B, L, D).

    Returns:
        tf.Tensor: Das Ergebnis der Attention-Berechnung der Form (B, L, H*D).
    """
    # Wenden Sie Rotary Position Embeddings (RoPE) auf die Abfrage und den Schlüssel an
    q, k = apply_rope(q, k, pe)
    
    # Berechne die skalierte dot-Produkt-Attention
    x = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(k.shape[-1])))
    x = tf.matmul(x, v)
    
    # Rearrange (transformiere) die Ausgabe in die Form (B, L, H*D)
    x = einops.rearrange(x, "B H L D -> B L (H D)")
    
    return x

def rope(
        pos: tf.Tensor,
        dim: int, 
        theta: int
) -> tf.Tensor:
    """
    Berechnet Rotary Positional Embeddings (RoPE) basierend auf den gegebenen Positionen und Dimensionen.

    Args:
        pos (tf.Tensor): Positionstensor der Form (B, L).
        dim (int): Dimension der Einbettungen (muss gerade sein).
        theta (int): Frequenz-Skalierungsfaktor.

    Returns:
        tf.Tensor: Ein Tensor mit den berechneten RoPEs der Form (B, L, D, 2, 2).
    """
    # Sicherstellen, dass die Dimension gerade ist
    assert dim % 2 == 0
    
    # Berechne den Skalierungsfaktor für die Winkel-Frequenzen
    scale = tf.range(0, dim, 2, dtype=tf.float64, name="scale") / dim
    omega = 1.0 / (theta ** scale)
    
    # Berechne die Positionseinbettungen
    out = tf.einsum("...n,d->...nd", pos, omega)
    
    # Berechne die sinusförmigen und kosinusförmigen Komponenten
    out = tf.stack([tf.cos(out), -tf.sin(out), tf.sin(out), tf.cos(out)], axis=-1)
    
    # Rearrange die Ausgabe in die Form (B, L, D, 2, 2)
    out = einops.rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    
    return tf.cast(out, tf.float32)

def apply_rope(
        xq: tf.Tensor, 
        xk: tf.Tensor, 
        freqs_cis: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Wendet Rotary Position Embeddings (RoPE) auf Abfrage- und Schlüssel-Tensoren an.

    Args:
        xq (tf.Tensor): Abfrage-Tensor der Form (B, H, L, D).
        xk (tf.Tensor): Schlüssel-Tensor der Form (B, H, L, D).
        freqs_cis (tf.Tensor): Frequenz- und Kosinus-Tensor der Form (B, L, D, 2, 2).

    Returns:
        tuple[tf.Tensor, tf.Tensor]: Die transformierten Abfrage- und Schlüssel-Tensoren.
    """
    # Transformiere die Abfrage- und Schlüssel-Tensoren in die richtige Form
    xq_ = tf.cast(xq, tf.float32)
    xk_ = tf.cast(xk, tf.float32)
    xq_ = tf.reshape(xq_, (*xq_.shape[:-1], -1, 1, 2))
    xk_ = tf.reshape(xk_, (*xk_.shape[:-1], -1, 1, 2))
    
    # Wende RoPE auf die Abfrage- und Schlüssel-Tensoren an
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    
    # Rücktransformation in die ursprüngliche Form
    xq_out = tf.reshape(xq_out, xq.shape)
    xk_out = tf.reshape(xk_out, xk.shape)
    
    return tf.cast(xq_out, xq.dtype), tf.cast(xk_out, xk.dtype)
