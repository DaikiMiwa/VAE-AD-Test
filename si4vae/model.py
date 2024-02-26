import fire
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from si4dnn import layers


def get_auto_encoder(shape=(16, 16, 1)):

    inputs = tf.keras.layers.Input(shape=shape)

    # Encoder
    conv1 = tf.keras.layers.Conv2D(4, (3, 3), activation="relu", padding="same")(inputs)
    conv2 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(conv1)
    mp1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(mp1)
    conv4 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(conv3)

    # Decoder
    conv5 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(conv4)
    conv6 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(conv5)
    up1 = tf.keras.layers.UpSampling2D((2, 2))(conv6)
    conv7 = tf.keras.layers.Conv2D(4, (3, 3), activation="relu", padding="same")(up1)
    conv8 = tf.keras.layers.Conv2D(1, (3, 3), activation="linear", padding="same")(
        conv7
    )

    model = tf.keras.models.Model(inputs=inputs, outputs=conv8)

    return model

def train_variational_auto_encoder(size: int,X_train=None,X_valid=None, output_dir=None, latent_dim = 10):

    np.random.seed(42)
    shape = (size, size, 1)

    if output_dir is None:
        output_dir = f"./trained_models/test_vae_model_{size}.h5",
    else :
        output_dir = f"{output_dir}/vae_model_{size}.h5"

    if X_train is None and X_valid is None:
        X_train = np.random.rand(1000, shape[0], shape[1], shape[2])
        X_valid = np.random.rand(100, shape[0], shape[1], shape[2])

    model = get_variational_auto_encoder(shape, latent_dim=latent_dim)
    model.summary()

    model_checkpoint = ModelCheckpoint(
        output_dir,
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )

    model.compile(optimizer="adam")
    model.fit(
        X_train,
        X_train,
        epochs=300,
        batch_size=16,
        validation_data=(X_valid, X_valid),
        callbacks=[model_checkpoint],
    )

def get_variational_auto_encoder(shape=(16, 16, 1), latent_dim=20):

    inputs = tf.keras.layers.Input(shape=shape)

    # Encoder
    conv1 = tf.keras.layers.Conv2D(4, (3, 3), activation="relu", padding="same")(inputs)
    conv2 = tf.keras.layers.Conv2D(4, (3, 3), activation="relu", padding="same")(conv1)
    mp1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(mp1)
    conv4 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(conv3)

    latent_image_shape = conv4.shape

    flatten = tf.keras.layers.Flatten()(conv4)

    # encoder output(latent mean and variance)
    z_mean = tf.keras.layers.Dense(latent_dim)(flatten)
    z_std = tf.keras.layers.Dense(latent_dim, activation="relu")(flatten)

    # Sampling Latent Variable
    z = SamplingLayer()([z_mean, z_std])
    z_input = tf.keras.layers.Dense(
        latent_image_shape[1]
        * latent_image_shape[2]
        * latent_image_shape[3]
    )(z)
    z_input = tf.keras.layers.Reshape(latent_image_shape[1:])(z_input)

    # Decoder
    conv5 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(
        z_input
    )
    conv6 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(conv5)
    up1 = tf.keras.layers.UpSampling2D((2, 2))(conv6)
    conv7 = tf.keras.layers.Conv2D(4, (3, 3), activation="relu", padding="same")(up1)
    conv8 = tf.keras.layers.Conv2D(4, (3, 3), activation="relu", padding="same")(conv7)
    conv9 = tf.keras.layers.Conv2D(1, (3, 3), activation="linear", padding="same")(
        conv8
    )

    vae = tf.keras.Model(inputs=inputs, outputs=[conv9,z_mean,z_std])
    vae.use_kl_loss = True

    reconstruction_loss = vae_reconstruction_loss(inputs,conv9)
    kl_loss = vae_kl_loss(z_mean,z_std)

    if vae.use_kl_loss:
        loss = kl_loss + reconstruction_loss
    else:
        loss = reconstruction_loss 

    vae.add_loss(loss)
    vae.add_metric(reconstruction_loss,name="reconstruction_loss")
    vae.add_metric(kl_loss,name="kl_loss")

    return vae

def get_normal_variational_auto_encoder(shape=(16, 16, 1), latent_dim=10):

    inputs = tf.keras.layers.Input(shape=shape)

    # Encoder
    conv1 = tf.keras.layers.Conv2D(4, (3, 3), activation="relu", padding="same")(inputs)
    conv2 = tf.keras.layers.Conv2D(4, (3, 3), activation="relu", padding="same")(conv1)
    mp1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(mp1)
    conv4 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(conv3)

    latent_image_shape = conv4.shape

    flatten = tf.keras.layers.Flatten()(conv4)

    # encoder output(latent mean and variance)
    z_mean = tf.keras.layers.Dense(latent_dim)(flatten)
    z_var = tf.keras.layers.Dense(latent_dim, activation="relu")(flatten)

    # Sampling Latent Variable
    z = NormalSamplingLayer()([z_mean, z_var])

    z_input = tf.keras.layers.Dense(
        latent_image_shape[1]
        * latent_image_shape[2]
        * latent_image_shape[3]
    )(z)
    z_input = tf.keras.layers.Reshape(latent_image_shape[1:])(z_input)

    # Decoder
    conv5 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(
        z_input
    )
    conv6 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(conv5)
    up1 = tf.keras.layers.UpSampling2D((2, 2))(conv6)
    conv7 = tf.keras.layers.Conv2D(4, (3, 3), activation="relu", padding="same")(up1)
    conv8 = tf.keras.layers.Conv2D(1, (3, 3), activation="linear", padding="same")(
        conv7
    )

    vae = tf.keras.Model(inputs=inputs, outputs=[conv8,z_mean,z_var])

    temp_reconstruction_loss = vae_reconstruction_loss(inputs,conv8)
    loss = normal_vae_kl_loss(z_mean, z_var)+ temp_reconstruction_loss

    kl_loss = normal_vae_kl_loss(z_mean,z_var)
    reconstruction_loss = vae_reconstruction_loss(inputs,conv8)

    vae.add_loss(loss)
    vae.add_metric(reconstruction_loss,name="reconstruction_loss")
    vae.add_metric(kl_loss,name="kl_loss")

    return vae

class SamplingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SamplingLayer,self).__init__()
        self.do_sample=True

    def build(self,input_shape):
        pass

    def call(self, inputs):
        latent_mean, latent_std = inputs
        latent_dim = latent_mean.shape[-1]

        epsilon = tf.keras.backend.random_normal(shape=(latent_dim,))

        if self.do_sample:
            output = latent_mean + epsilon * latent_std
        else: 
            output = latent_mean

        return output

class NormalSamplingLayer(tf.keras.layers.Layer):
    def __call__(self, inputs):
        latent_mean, latent_std = inputs
        latent_dim = latent_mean.shape[-1]

        epsilon = tf.keras.backend.random_normal(shape=(latent_dim,))
        output = latent_mean + epsilon * tf.keras.backend.exp(0.5 * latent_std)

        return output


class SampligSILayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs):
        latent_mean, latent_std = inputs
        latent_dim = latent_mean.shape[-1]

        epsilon = tf.keras.backend.random_normal(shape=(latent_dim,))
        self.epsilon = tf.cast(epsilon,dtype=tf.float64)

        output = latent_mean + epsilon * latent_std

        return output
    def forward_si(self,x,bias,a,b,l,u):

        output_x = x[0] + self.epsilon * x[1]
        output_bias = bias[0] + self.epsilon * bias[1]
        output_a = a[0] + self.epsilon * a[1]
        output_b = b[0] + self.epsilon * b[1]

        return output_x,output_bias,output_a,output_b,l,u

def vae_batch_reconstruction_loss(y_true, y_predict):
    reconstruction_loss_factor = 1000
    reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_predict[0]), axis=[1, 2, 3])

    return reconstruction_loss_factor * reconstruction_loss

def vae_batch_kl_loss(y_true, y_predict):
    encoder_mu = y_predict[1]
    encoder_std = y_predict[2]
    encoder_std += 1e-6
    array = (1.0 + 2.0 * tf.keras.backend.log(encoder_std) - tf.keras.backend.square(encoder_mu) - tf.keras.backend.square(encoder_std))
    kl_loss = -0.5 * tf.keras.backend.sum(array, axis=1)

    return kl_loss

def vae_reconstruction_loss(y_true, y_predict):
    reconstruction_loss_factor = 1000
    reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_predict), axis=[1, 2, 3])

    return reconstruction_loss_factor * reconstruction_loss

def vae_kl_loss(encoder_mu, encoder_std):
    encoder_std += 1e-6
    array = (1.0 + 2.0 * tf.keras.backend.log(encoder_std) - tf.keras.backend.square(encoder_mu) - tf.keras.backend.square(encoder_std))
    kl_loss = -0.5 * tf.keras.backend.sum(array, axis=1)

    return kl_loss

def normal_vae_kl_loss(encoder_mu, encoder_log_var):
    kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_var - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_var), axis=1)

    return kl_loss

def vae_kl_loss_metric(y_true, y_predict):
    encoder_mu = y_predict[1]
    encoder_std = y_predict[2]

    kl_loss = -0.5 * tf.keras.backend.sum(1.0 + tf.keras.backend.square(encoder_std) + tf.keras.backend.square(encoder_mu) - tf.keras.backend.square(encoder_std), axis=1)
    return kl_loss

def vae_loss(y_true, y_predict):
    reconstruction_loss = vae_reconstruction_loss(y_true,  y_predict[0])
    kl_loss = vae_kl_loss(y_predict[1], y_predict[2])

    loss = reconstruction_loss + kl_loss
    return loss

def train_normal_variational_auto_encoder(size: int):
    np.random.seed(42)
    shape = (size, size, 1)

    X_train = np.random.rand(1000, shape[0], shape[1], shape[2])
    X_valid = np.random.rand(100, shape[0], shape[1], shape[2])

    model = get_normal_variational_auto_encoder(shape)
    model.summary()

    model_checkpoint = ModelCheckpoint(
        f"./trained_models/test_vae_model_{size}.h5",
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )

    model.compile(optimizer="adam")
    model.fit(
        X_train,
        X_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_valid, X_valid),
        callbacks=[model_checkpoint],
    )


def train_auto_encoder(size: int):
    np.random.seed(42)

    shape = (size, size, 1)

    X_train = np.random.rand(1000, shape[0], shape[1], shape[2])
    X_valid = np.random.rand(100, shape[0], shape[1], shape[2])

    model = model = get_auto_encoder(shape)

    model_checkpoint = ModelCheckpoint(
        f"./trained_models/test_model_{size}.h5",
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(
        X_train,
        X_train,
        epochs=10,
        batch_size=16,
        validation_data=(X_valid, X_valid),
        callbacks=[model_checkpoint],
    )


def reconstruct(model, X):
    reconstruct = model.predict(X)
    print(reconstruct.shape)

    return reconstruct


def show_reconstruct_error(X, X_reconstruct, thr):
    plt.rcParams["font.size"] = 18
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    ax1.set_title("Original")
    ax1.imshow(X[0, :, :, 0])

    ax2.set_title("Reconstruct")
    ax2.imshow(X_reconstruct[0, :, :, 0])

    ax3.set_title("Reconstruct Error")
    ax3.imshow(X[0, :, :, 0] - X_reconstruct[0, :, :, 0])

    ax4.set_title("Abnomal Area")
    ax4.imshow(X[0, :, :, 0] - X_reconstruct[0, :, :, 0] > thr)


if __name__ == "__main__":
    fire.Fire(train_auto_encoder)
