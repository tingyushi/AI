import tensorflow as tf
import tensorflow.keras.layers as tfl
import tensorflow.keras.models as tfm
from hparams import *

# generator
def get_generator(input_dim):

    # input
    input = tf.keras.Input(input_dim)

    # from input_dim -> 7x7xinput_dim
    x = tfl.Dense(units = 7 * 7 * input_dim)(input)
    x = tfl.LeakyReLU(0.2)(x)
    x = tfl.Reshape((7, 7, input_dim))(x)

    # upsampling
    # from 7x7xinput_dim -> 14x14x128
    x = tfl.UpSampling2D()(x)
    x = tfl.Conv2D(128, 5, padding = 'same')(x)
    x = tfl.LeakyReLU(0.2)(x)

    # upsampling
    # from 14x14x128 -> 28x28x128
    x = tfl.UpSampling2D()(x)
    x = tfl.Conv2D(128, 5, padding = 'same')(x)
    x = tfl.LeakyReLU(0.2)(x)

    # conv
    x = tfl.Conv2D(128, 4, padding='same')(x)
    x = tfl.LeakyReLU(0.2)(x)
    x = tfl.Conv2D(128, 4, padding='same')(x)
    x = tfl.LeakyReLU(0.2)(x)
    x = tfl.Conv2D(1, 4, padding = 'same', activation = 'sigmoid')(x)

    # output
    output = x
    
    model = tfm.Model(inputs = input, outputs = output)
    
    return model


# discriminator
def get_discriminator(input_dim):
    
    input = tf.keras.Input(input_dim)

    x = tfl.Conv2D(32, 5, padding = 'valid')(input)
    x = tfl.LeakyReLU(0.2)(x)
    x = tfl.Dropout(0.4)(x)

    x = tfl.Conv2D(64, 5, padding = 'valid')(x)
    x = tfl.LeakyReLU(0.2)(x)
    x = tfl.Dropout(0.4)(x)


    x = tfl.Conv2D(128, 5, padding = 'valid')(x)
    x = tfl.LeakyReLU(0.2)(x)
    x = tfl.Dropout(0.4)(x)

    x = tfl.Conv2D(256, 5, padding = 'valid')(x)
    x = tfl.LeakyReLU(0.2)(x)
    x = tfl.Dropout(0.4)(x)

    x = tfl.Flatten()(x)
    x = tfl.Dropout(0.4)(x)
    
    output = tfl.Dense(units = 1, activation = 'sigmoid')(x)
    
    model = tfm.Model(inputs = input, outputs = output)
    
    return model


# real image -> 0
# fake image -> 1
class GAN(tfm.Model):
    def __init__(self, generator, discriminator, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen = generator
        self.dis = discriminator
        self.latent_dim = latent_dim

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs): 
        super().compile(*args, **kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss 

    def train_step(self, image_batch):
        batch_size, height, width, channel = image_batch.shape
        
        # get real and fake images
        real_images = image_batch
        fake_images = self.gen(tf.random.normal((batch_size, self.latent_dim, 1)),  training=False)

        # train dis
        with tf.GradientTape() as d_tape:

            # make predictions
            yhat_real = self.dis(real_images, training = True)
            yhat_fake = self.dis(fake_images, training = True)
            yhat = tf.concat([yhat_real, yhat_fake], axis = 0)

            # correct labels
            y = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis = 0)

            # Add some noise to the correct labels
            noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(yhat_fake))
            y += tf.concat([noise_real, noise_fake], axis=0)
            
            # Calculate loss - BINARYCROSS 
            dis_loss = self.d_loss(y, yhat)

        # apply gradient
        dgrad = d_tape .gradient(dis_loss, self.dis.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.dis.trainable_variables))

        # train gan
        with tf.GradientTape() as g_tape:
            # generate fake images
            fake_images = self.gen(tf.random.normal((batch_size, self.latent_dim, 1)),  training=True)

            # get predicted labels
            yhat = self.dis(fake_images, training = False)

            # wished labels, we want fake images to be real images
            wished_labels = tf.zeros_like(yhat)

            gen_loss = self.g_loss(yhat, wished_labels)

        # apply gradient
        ggrad = g_tape .gradient(gen_loss, self.gen.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.gen.trainable_variables))
        
        return {"dis_loss": dis_loss, "gen_loss": gen_loss}