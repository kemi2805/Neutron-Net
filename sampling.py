import math
import numpy as np
import tensorflow as tf

from typing import Callable



def get_noise(
        num_samples: int,
        height: int,
        width: int,
        dtype: tf.dtypes.DType,
        seed: int
) -> tf.Tensor:
    """Generate a tensor of random noise with a normal distribution.

    Args:
        num_samples (int): Number of samples (batch size).
        height (int): Height of the noise tensor.
        width (int): Width of the noise tensor.
        dtype (tf.dtypes.DType): The data type of the tensor (e.g., tf.float32).
        seed (int): Seed for random number generation.

    Returns:
        tf.Tensor: Tensor of random noise.
    """
    # Calculate padded dimensions
    padded_height = 2 * math.ceil(height / 16)
    padded_width = 2 * math.ceil(width / 16)

    # Create a random normal distribution
    noise = tf.random.normal(
        shape=[num_samples, 16, padded_height, padded_width],
        mean=0.0,
        stddev=1.0,
        dtype=dtype,
        seed=seed
    )

    return noise
    
def time_shift(mu: float, sigma: float, t: tf.Tensor) -> tf.Tensor:
    """Applies a time shift to the tensor t based on mu and sigma.

    Args:
        mu (float): The shift parameter.
        sigma (float): The shape control parameter.
        t (tf.Tensor): A tensor of time steps.

    Returns:
        tf.Tensor: A tensor with time steps shifted according to mu and sigma.
    """
    # Apply the time shift formula
    return tf.math.exp(mu) / (tf.math.exp(mu) + (1 / t - 1) ** sigma)

    
def get_linear_function(
        x1: float = 256, 
        y1: float = 0.5, 
        x2: float = 4096, 
        y2: float = 1.15
) -> Callable[[float], float]:
    """Generate a linear function that maps x to y based on two points.

    Args:
        x1 (float): x-coordinate of the first point.
        y1 (float): y-coordinate of the first point.
        x2 (float): x-coordinate of the second point.
        y2 (float): y-coordinate of the second point.

    Returns:
        Callable[[float], float]: A function that computes the linear interpolation.
    """
    def linear_function(x: float) -> float:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m * x + b
    
    return linear_function

def get_cosine_function(
        x1: float = 256, 
        y1: float = 0.5, 
        x2: float = 4096, 
        y2: float = 1.15
) -> Callable[[float], float]:
    """Generate a cosine interpolation function that maps x to y based on two points.

    Args:
        x1 (float): x-coordinate of the first point.
        y1 (float): y-coordinate of the first point.
        x2 (float): x-coordinate of the second point.
        y2 (float): y-coordinate of the second point.

    Returns:
        Callable[[float], float]: A function that computes the cosine interpolation.
    """
    def cosine_function(x: float) -> float:
        # Normalize x between x1 and x2
        normalized_x = (x - x1) / (x2 - x1)
        # Apply the cosine interpolation
        return y1 + (y2 - y1) / 2 * (1 - math.cos(math.pi * normalized_x))
    
    return cosine_function


def get_schedule(
        num_steps: int,
        image_seq_len: int,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift: bool = True
) -> list[float]:
    """Generate a schedule of time steps with optional shifting.

    Args:
        num_steps (int): Number of time steps to generate.
        image_seq_len (int): Length of the image sequence (used for shifting).
        base_shift (float): Base shift value.
        max_shift (float): Maximum shift value.
        shift (bool): Whether to apply shifting to the time steps.

    Returns:
        list[float]: list of time steps, potentially shifted.
    """
    # Generate time steps from 1.0 to 0.0
    timesteps = tf.linspace(1.0, 0.0, num_steps + 1)
    
    if shift:
        # Get the linear function for shifting
        lin_function = get_linear_function(y1=base_shift, y2=max_shift)
        mu = lin_function(image_seq_len)
        
        # Apply time shifting
        timesteps = time_shift(mu, 1.0, timesteps)
    
    # Convert tensor to list
    return timesteps.numpy().tolist()

def denoise(
    model: tf.Module,  # Changed to tf.Module for flexibility
    img: tf.Tensor,
    img_ids: tf.Tensor,
    vec: tf.Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
) -> tf.Tensor:
    """Apply iterative denoising to an image tensor using a model and guidance.

    Args:
        model (tf.Module): A TensorFlow model that predicts noise given inputs.
        img (tf.Tensor): The input image tensor to denoise.
        img_ids (tf.Tensor): Image identifiers or associated data.
        vec (tf.Tensor): Additional vector input, such as a latent vector or embedding.
        timesteps (list[float]): A list of time steps for the denoising process.
        guidance (float): A scalar controlling the influence of guidance during denoising.

    Returns:
        tf.Tensor: The denoised image tensor.
    """
    # Create a guidance vector filled with the guidance value, matching the batch size
    guidance_vec = tf.fill([img.shape[0]], guidance, name='guidance_vec')
    
    # Iterate over the time steps in reverse order (from t_curr to t_prev)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        # Create a vector filled with the current time step, matching the batch size
        t_vec = tf.fill([img.shape[0]], t_curr, name='t_vec')
        
        # Predict noise using the model
        pred = model(
            img=img,
            img_ids=img_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        # Update the image tensor by adjusting it based on the predicted noise
        img = img + (t_prev - t_curr) * pred

    # Return the final denoised image tensor
    return img



# Eher nicht anwendbar für mich
def unpack(x: tf.Tensor, height: int, width: int) -> tf.Tensor:
    """Unpacks and reshapes a tensor based on height and width parameters.

    Args:
        x (tf.Tensor): The input tensor of shape [batch_size, ..., channels*16].
        height (int): The target height for the unpacked tensor.
        width (int): The target width for the unpacked tensor.

    Returns:
        tf.Tensor: The unpacked and reshaped tensor.
    """
    # Calculate dimensions
    batch_size = tf.shape(x)[0]  # Get the dynamic batch size
    channels = tf.shape(x)[-1] // 16  # Original channels, assuming last dim is channels * 16
    new_height = math.ceil(height / 16)  # Calculate the new height after unpacking
    new_width = math.ceil(width / 16)  # Calculate the new width after unpacking

    # Reshape the tensor to isolate the channels
    reshaped_x = tf.reshape(x, [batch_size, -1, channels])

    # Transpose the tensor to prepare for reshaping to height and width
    unpacked = tf.transpose(reshaped_x, perm=[0, 2, 1])

    # Reshape the tensor to the target shape [batch_size, height, width, channels]
    unpacked = tf.reshape(unpacked, [-1, new_height * 2, new_width * 2, channels])

    return unpacked

