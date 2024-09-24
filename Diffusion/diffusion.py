import tensorflow as tf
from keras.models import Model
import numpy as np
# Ich schreibe hier einfach mal zahlreiche Funktionen welche ich gebrauchen könnte
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


@tf.function
def forward_diffusion_step(image, beta_schedule, t):
    noise = tf.random.normal(shape=image.shape)
    alpha = 1 - beta_schedule[t]
    alpha_bar = tf.math.cumprod([alpha], axis=0)
    image_noisy = tf.sqrt(alpha_bar) * image + tf.sqrt(1 - alpha_bar) * noise
    return image_noisy

@tf.function
def reverse_diffusion_step(model, image_noisy, beta_schedule, t):
    noise_pred = model(image_noisy, training=True)
    loss = tf.reduce_mean(tf.square(noise_pred))
    return loss

@tf.function
def train_step(model, images, beta_schedule, t, accumulation_steps, optimizer):
    accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]

    for _ in range(accumulation_steps):
        with tf.GradientTape() as tape:
            image_noisy = forward_diffusion_step(images, beta_schedule, t)
            loss = reverse_diffusion_step(image_noisy, beta_schedule, t)
        gradients = tape.gradient(loss, model.trainable_variables)
        accumulated_gradients = [accum_grad + grad for accum_grad, grad in zip(accumulated_gradients, gradients)]

    # Apply accumulated gradients
    accumulated_gradients = [grad / accumulation_steps for grad in accumulated_gradients]
    optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))

    return loss
