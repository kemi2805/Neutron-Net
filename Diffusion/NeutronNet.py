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
        noise = tf.random.normal(shape=image.shape, dtype=tf.float64)
        alpha = 1 - self.beta_schedule[t]
        alpha_bar = tf.math.cumprod(tf.cast([1.0] + [alpha], tf.float64), axis=0)[-1]
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
        #loss = tf.reduce_mean(tf.square(noise_pred))
        return noise_pred
    
    @tf.function
    def validation_step(
        self, 
        images: tf.Tensor, 
        t: int, 
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]=tf.keras.losses.MeanSquaredError()
    ) -> tf.Tensor:
        """
        Performs validation, including gradient accumulation.

        Args:
            images (tf.Tensor): A batch of image tensors for training.
            t (int): The current time step in the diffusion process.
            optimizer (tf.keras.optimizers.Optimizer): The optimizer to update the model weights.
            loss_fn (Callable): The loss function

        Returns:
            tf.Tensor: The computed loss for validation data.
        """
        image_noisy = self.forward_diffusion_step(images, t)
        noise_pred = self.reverse_diffusion_step(image_noisy, t)
        loss = loss_fn(image_noisy, noise_pred)
        return loss
    
    #@tf.function
    def train_step(
        self, 
        images: tf.Tensor, 
        t: int, 
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]=tf.keras.losses.MeanSquaredError()
    ) -> tf.Tensor:
        """
        Performs a single training step, including gradient accumulation.

        Args:
            images (tf.Tensor): A batch of image tensors for training.
            t (int): The current time step in the diffusion process.
            optimizer (tf.keras.optimizers.Optimizer): The optimizer to update the model weights.
            loss_fn (Callable): The loss function of the neural net

        Returns:
            tf.Tensor: The computed loss for the current training step.
        """
        accumulated_gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
        total_loss = 0.0

        #sub_batch_size = images.shape[0] // self.accumulation_steps
        sub_batch_size = tf.shape(images)[0] // self.accumulation_steps
        for i in range(self.accumulation_steps):
            sub_batch = images[int(i * sub_batch_size):int((i + 1) * sub_batch_size)]
            with tf.GradientTape() as tape:
                image_noisy = self.forward_diffusion_step(sub_batch, t)
                predicted_images = self.reverse_diffusion_step(image_noisy, t)
                # Compute loss using the dynamic loss function
                loss = loss_fn(predicted_images, sub_batch)  # Use loss_fn dynamically
                total_loss += loss

            gradients = tape.gradient(loss, self.model.trainable_variables)
            accumulated_gradients = [accum_grad + grad for accum_grad, grad in zip(accumulated_gradients, gradients)]

        # Apply accumulated gradients
        accumulated_gradients = [grad / self.accumulation_steps for grad in accumulated_gradients]
        optimizer.apply_gradients(zip(accumulated_gradients, self.model.trainable_variables))
        total_loss = total_loss.numpy()

        return total_loss / tf.cast(self.accumulation_steps, tf.float32)

    #@tf.function
    def train_model(
        self, 
        train_dataset: tf.data.Dataset, 
        num_epochs: int, 
        callbacks: List, 
        optimizer, 
        val_dataset: tf.data.Dataset=None,
        batch_size: int = 1,
        loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]=tf.keras.losses.MeanSquaredError(),
        metric_fn: Callable[[tf.Tensor], tf.Tensor]=tf.keras.metrics.Mean()
    ) -> None:
        """
        Performs the model training.

        Args:
            dataset (tf.data.Dataset): The training dataset (batched, prefetched, etc.).
            num_epochs (int): Number of epochs to train for.
            callbacks (List): A list of callbacks to be used.
            optimizer (tf.keras.optimizers.Optimizer): The optimizer to update the model weights.
            validation_data (tf.data.Dataset): Validation dataset, optional.
            batch_size (int): Size of batches
            loss_fn (Callable): The loss function of the neural net
            metric_fn (Callable): The metric function for logging

        Returns:
            None
        """
        # Notify callbacks that training is beginning
        for callback in callbacks:
            callback.on_train_begin()

        step = -1
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Notify callbacks that a new epoch is beginning
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            
            # Training loop
            train_loss = loss_fn
            train_metric = metric_fn
            for images in train_dataset:
                step += 1
                #t = tf.random.uniform([], minval=0, maxval=len(self.beta_schedule), dtype=tf.int32)
                t = tf.random.uniform([], minval=0, maxval=tf.shape(self.beta_schedule)[0], dtype=tf.int32)
                t = int(t.numpy())
                step_loss = self.train_step(images, t, optimizer, train_loss)
                train_metric.update_state(step_loss)

                # Call callbacks for each batch
                for callback in callbacks:
                    callback.on_train_batch_end(step, logs={'loss': step_loss})

                if step % 100 == 0:
                    tf.print(f"Step {step}, Loss: {step_loss:.4f}")

            # Validation loop
            if val_dataset:
                val_metric = metric_fn
                for val_images in val_dataset:
                    print(val_images)
                    #t = tf.random.uniform([], minval=0, maxval=len(self.beta_schedule), dtype=tf.int32)
                    t = tf.random.uniform([], minval=0, maxval=tf.shape(self.beta_schedule)[0], dtype=tf.int32)
                    t = int(t.numpy())
                    val_step_loss = self.validation_step(val_images, t, optimizer)  # oder wie du val_loss berechnen möchtest
                    val_metric.update_state(val_step_loss)
                #if len(val_dataset) != 0:
                #    val_metric /= len(val_dataset)

            # Epoch end
            logs = {'loss': train_metric.result()}
            if val_dataset:
                logs['val_loss'] = val_metric.result()

            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=logs)

            tf.print(f"Epoch {epoch + 1} finished. Train Loss: {logs['loss']:.4f}" + 
                (f", Val Loss: {logs['val_loss']:.4f}" if val_dataset else ""))

            train_metric.reset_states()
            if val_dataset:
                val_metric.reset_states()

            for callback in callbacks:
                callback.on_train_end()





class CosineBetaScheduler:
    def __init__(self, num_timesteps: int, s: float = 0.008):
        self.num_timesteps = num_timesteps
        self.s = s
        
        # Generate cosine schedule
        self.cosine_schedule = self._generate_cosine_schedule()

    def _generate_cosine_schedule(self) -> tf.Tensor:
        x = tf.linspace(0, 1, self.num_timesteps)
        alpha = tf.cos(((x+self.s) / (1+self.s) * (pi / 2)))
        alpha = tf.square(alpha)
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
        alpha_0 = self.cosine_schedule[0]
        alpha_t = self.cosine_schedule[t]
        beta_t = 1 - alpha_t/alpha_0
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
    
    def get_schedule(self) -> tf.Tensor:
        alpha_0 = self.cosine_schedule[0]
        alpha_t = self.cosine_schedule
        beta_t = 1 - alpha_t/alpha_0
        return beta_t

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
