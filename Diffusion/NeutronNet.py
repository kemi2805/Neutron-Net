import tensorflow as tf
from autoencoder import AutoEncoder, AutoEncoderParams
from numpy import pi
from typing import Callable, List

class NeutronNet(tf.keras.Model):
    def __init__(
            self, 
            AutoEncoderParams: AutoEncoderParams,
            beta_schedule: tf.Tensor,
            accumulation_steps: int=4
    ):
        """
        Initializes the NeutronNet model.

        Args:
            autoencoder_params (AutoEncoderParams): Parameters for the AutoEncoder.
            beta_schedule (tf.Tensor): A tensor representing the beta schedule for noise.
            accumulation_steps (int): Number of steps for gradient accumulation.
        """
        super(NeutronNet, self).__init__()
        self.model = self.build_model(AutoEncoderParams)
        self.beta_schedule = beta_schedule
        self.accumulation_steps = accumulation_steps

    def build_model(
            self, 
            autoencoder_params: AutoEncoderParams
    ) -> AutoEncoder:
        model = AutoEncoder(params=autoencoder_params)
        return model

    def set_beta_schedule(
            self, 
            new_beta_schedule: tf.Tensor
    ) -> None:
        self.beta_schedule = new_beta_schedule

    def set_accumulation_steps(
            self, 
            new_accumulation_steps: int
    ) -> None:
        self.accumulation_steps = new_accumulation_steps

    def set_model(
            self, 
            new_model: tf.keras.Model
    ) -> None:
        self.model = new_model

    @tf.function
    def forward_diffusion_step(
        self, 
        image: tf.Tensor,
        t: int
    ) -> tf.Tensor:
        """
        Performs the forward diffusion step by adding noise to the image.

        Args:
            image (tf.Tensor): The original image tensor.
            t (int): The current time step in the diffusion process.

        Returns:
            tf.Tensor: The noisy image after applying the diffusion process.
        """
        noise = tf.random.normal(shape=image.shape)
        alpha = 1 - self.beta_schedule[t]
        alpha_bar = tf.math.cumprod([1] + [alpha], axis=0)[-1]
        image_noisy = tf.sqrt(alpha_bar) * image + tf.sqrt(1 - alpha_bar) * noise
        return image_noisy

    @tf.function
    def reverse_diffusion_step(
        self, 
        image_noisy: tf.Tensor, 
        t: int
    ) -> tf.Tensor:
        """
        Performs the reverse diffusion step to predict the noise from the noisy image.

        Args:
            image_noisy (tf.Tensor): The noisy image tensor.
            t (int): The current time step in the diffusion process.

        Returns:
            tf.Tensor: The calculated loss between predicted noise and actual noise.
        """
        noise_pred = self.model(image_noisy, training=True)
        loss = tf.reduce_mean(tf.square(noise_pred))
        return loss

    @tf.function
    def train_step(
        self, 
        images: tf.Tensor, 
        t: int, 
        optimizer: tf.keras.optimizers.Optimizer
    ) -> tf.Tensor:
        """
        Performs a single training step, including gradient accumulation.

        Args:
            images (tf.Tensor): A batch of image tensors for training.
            t (int): The current time step in the diffusion process.
            optimizer (tf.keras.optimizers.Optimizer): The optimizer to update the model weights.

        Returns:
            tf.Tensor: The computed loss for the current training step.
        """
        accumulated_gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]

        for _ in range(self.accumulation_steps):
            with tf.GradientTape() as tape:
                image_noisy = self.forward_diffusion_step(images, t)
                loss = self.reverse_diffusion_step(image_noisy, t)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            accumulated_gradients = [accum_grad + grad for accum_grad, grad in zip(accumulated_gradients, gradients)]

        # Apply accumulated gradients
        accumulated_gradients = [grad / self.accumulation_steps for grad in accumulated_gradients]
        optimizer.apply_gradients(zip(accumulated_gradients, self.model.trainable_variables))

        return loss

    @tf.function
    def train_model(
        self, 
        dataset: tf.Tensor, 
        num_epochs: int, 
        callbacks: List, 
        optimizer, 
        validation_data=None
    ) -> None:
        """
        Performs the model training.

        Args:
            dataset (tf.Tensor): The entiretry of data.
            num_epochs (int): The current time step in the diffusion process.
            callbacks (List): A list of callbacks to be used
            optimizer (tf.keras.optimizers.Optimizer): The optimizer to update the model weights.
            validation_data (tf.Tensor): Validation data

        Returns:
            None
        """
        for epoch in range(num_epochs):
            for step, images in enumerate(dataset):
                t = tf.random.uniform([], minval=0, maxval=len(self.beta_schedule), dtype=tf.int32)
                loss = self.train_step(images, t, optimizer)

                # Callbacks aufrufen für jeden Batch
                for callback in callbacks:
                    callback.on_train_batch_end(step, logs={'loss': loss})

            # Am Ende der Epoche: berechne val_loss, falls validation_data gegeben ist
            val_loss = None
            if validation_data:
                val_images = next(iter(validation_data))  # Nehme ein Batch von Validierungsdaten
                t = tf.random.uniform([], minval=0, maxval=len(self.beta_schedule), dtype=tf.int32)
                val_loss = self.train_step(val_images, t, optimizer)  # oder wie du val_loss berechnen möchtest

            # Am Ende der Epoche
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs={'loss': loss, 'val_loss': val_loss})


class CosineBetaScheduler:
    def __init__(self, num_timesteps: int, s: float = 0.008):
        self.num_timesteps = num_timesteps
        self.s = s
        
        # Generate cosine schedule
        self.cosine_schedule = self._generate_cosine_schedule()

    def _generate_cosine_schedule(self) -> tf.Tensor:
        x = tf.linspace(0, 1, self.num_timesteps)
        alpha = tf.cos((x / self.s * (pi / 2)))
        alpha = tf.clip_by_value(alpha, clip_value_min=0, clip_value_max=0.9999)
        return alpha

    def get_beta(self, t: int) -> float:
        """
        Get beta value for a given timestep.
        
        Args:
            t (int): Current timestep
        
        Returns:
            float: Beta value for the current timestep
        """
        alpha_t = self.cosine_schedule[t]
        beta_t = 1 - alpha_t
        return beta_t.numpy()

    def schedule(self, model_output: tf.Tensor, sample: tf.Tensor, t: int) -> tf.Tensor:
        """
        Schedule function to update the sample based on model output.
        
        Args:
            model_output (tf.Tensor): Model output for the current timestep
            sample (tf.Tensor): Current sample being diffused
            t (int): Current timestep
        
        Returns:
            tf.Tensor: Updated sample
        """
        beta_t = self.get_beta(t)
        alpha_t = 1 - beta_t
        
        # Update sample
        sample = tf.math.add(sample * alpha_t, model_output * beta_t)
        
        return sample

def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> Callable[[int], float]:
    """
    Generate a cosine-based beta schedule.
    
    Args:
        num_timesteps (int): Number of time steps
        s (float): Small constant used in calculation (default: 0.008)
    
    Returns:
        Callable[[int], float]: Function to get beta value for a given timestep
    """
    x = tf.linspace(0, 1, num_timesteps)
    alpha = tf.cos((x / s * (pi / 2)))
    alpha = tf.clip_by_value(alpha, clip_value_min=0, clip_value_max=0.9999)
    
    def get_beta(t):
        return 1 - alpha[t]
    
    return get_beta


if __name__ == "__main__":
    # Usage example
    input_shape = (256, 256, 1)
    num_steps = 1000
    beta_schedule = tf.linspace(0.0001, 0.02, num_steps)  # Example beta schedule

    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, History
    patience = 10
    checkpoint_path = "/mnt/rafast/miler/some_model"
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience, min_delta=0.001)
    callbacks = [checkpoint_callback, reduce_lr, early_stopping]

    # Create an instance of the NeutronNet
    neutron_net = NeutronNet(input_shape, beta_schedule)

    # Define an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Example of a single training step (you can loop through epochs and batches as needed)
    images = tf.random.normal((64, 256, 256, 1))  # Example input
    t = 0  # Example time step

    neutron_net.train_model()
