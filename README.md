# Intro
An attempt at a user-friendly PyTorch implementation of a Generative Adverserial Network (GAN). The user only needs to supply the data (images) in order to train their own GAN, and parameters such as learning rates and optimizer properties can be tweaked by the user.

GANs are trained by letting two different models compete against each other. The *generator* model attempts to create images that the *discriminator* model struggles to distinguish from real images. During training, the discriminator learns by attempting to classify real images as real and fake images (created by the generator) as fake. The generator learns from the feedback provided by the discriminator on how real the fake images look. If the two models are trained properly, the generator should be able to create images that look like real images, without ever being exposed to the real data directly.

The default generator and discriminator is built for handling 128x128 RGB images. If the user wants to work with different image sizes, or just with different model architectures in general, the user can supply their own generator and/or discriminator when training the GAN. The Usage-section covers an example on how to use such custom-defined models.

Most of the testing of this package has been done in a Jupyter Notebook environment, so I highly recommend using this package in an equivalent environment.

# Installation
Install using ```pip``` with ```pip install git+https://github.com/hglad/easygan``` .

*Alternatively*: Clone this repository, navigate to the root folder containing ```setup.py``` and run ```python -m pip install .``` .

# Training a GAN
#### Using default parameters
With ```imgs``` provided as a list or NumPy array of images, the following code will create a class instance and train a GAN using the default parameters:
```python
import easygan as eg

# Resize images to 128x128, normalize pixel values and return as tensor
imgs_tn = eg.preprocess(imgs, h=128, w=128)

mygan = eg.GAN()          
mygan.train_gan(imgs_tn)       
```
By default, the models (generator and discriminator) will be saved to a folder inside ```models``` in your working directory. Some results will also be generated and saved to the folder ```results```, also placed in your working directory.

Training can be resumed by running ```train_gan``` again. Note that this will save a new set of models rather than overwriting the previous models.

#### Using other parameters
Parameters can be supplied to the ```train_gan``` function (a list of tweakable parameters is provided in the final section of this Readme), providing some control over the training procedure and more:
```python    
mygan.train_gan(imgs_tn, batch_size=16, epochs=50, DiffAugment=True)       
```

#### Using custom generators and discriminators
If you want to use a different generator and/or discriminator than the default ones, you can provide the models as an additional parameter to the ```train_gan``` function. Simply import the desired model (should be a class that is a subclass of torch.nn.Module) and pass it as a keyword argument:

```python    
from MyGenerator import MyGenerator
from MyDiscriminator import MyDiscriminator

mygan.train_gan(imgs_tn, custom_G=MyGenerator, custom_D=MyDiscriminator)       
```

If the provided models take any input arguments (such as ```base_channels``` in the default models), they can be set by passing them as a keyword argument in the ```train_gan``` call.

#### Loading trained models
In order to load a generator and discriminator, provide the name of the folder inside ```models``` that contains the models:

```python
folder = '2021-08-28-1345'    # name of folder inside models/
mygan.load_state(folder)     
```

#### Accessing the generator and generating images
The generator can be accessed with ```self.G```. The module also has a function ```generate_image``` that can be used to create an image. Below is an example showing how to synthesize images with the model and displaying it with the ```matplotlib``` module:

```python
import torch
import matplotlib.pyplot as plt
z = torch.randn(1, 100)          # input latent space
img = mygan.generate_image(z)    # generated 128x128x3 image

# Display the image
plt.imshow(img)
plt.show()
```

#### Example notebook
A notebook that demonstrates the usage of this package is provided in the ``easygan/easygan/example`` folder.

# Training a CGAN
Images and their corresponding class labels must be supplied in order to train a CGAN. For example, with ```labels``` as an array of integers (0, 1, ... , num_classes-1) and ```imgs``` containing the training images, the following code will train a CGAN using default parameters:

```python
import easygan as eg

# Resize images to 160x90, normalize pixel values and return as tensor
imgs_tn = eg.preprocess(imgs, h=90, w=160)

mygan = eg.CGAN()          
mygan.train_cgan(imgs_tn, labels, batch_size=16, epochs=50)       
```
The default generator and discriminator are only suitable for 160x90 images (width x height).

As with training a GAN, the models are automatically saved and training can be resumed later. When generating an image using a CGAN, the class label must also be given, for example:

```python
import torch
import matplotlib.pyplot as plt
z = torch.randn(1, 100)          # input latent space
y = torch.Tensor([2]).long()       # class label of desired image
img = mygan.generate_image(z, y)   

# Display the image
plt.imshow(img)
plt.show()
```



# Parameters
The user can tweak a variety of parameters that affect the training procedure, network architecture and more.

**batch_size**: How many images that are passed through the model at a time. Smaller batch sizes lead to slower training but can yield better results. **Default**: 128

**epochs**: Number of complete passes through the data to perform during training. **Default**: 100

**use_cuda**: Train model using GPU if enabled. **Default**: True

**lr_g**: Learning rate when optimizing generator. **Default**: 0.0002

**lr_d**: Learning rate when optimizing discriminator. **Default**: 0.0002

**beta1**: Parameter for the Adam optimizer. **Default**: 0.5

**beta2**: Parameter for the Adam optimizer. **Default**: 0.999

**shuffle**: If True, shuffle the data before feeding it into the models. **Default**: True      

**loss**: Which PyTorch loss function to use. **Default**: BCELoss   

**DiffAugment**: If True, use data augmentation methods provided by Zhao et al. in *Differentiable Augmentation for Data-Efficient GAN Training* (2020). Helpful when training on a small dataset. **Default**: False

**augment_policy**: If ```DIffAUgment``` is True, perform data augmentation with these methods (available options are color, translation, cutout). **Default**: 'color,translation'

**z_size**: Size of the input vector to the generator (the latent space vector). **Default**: 100    

**base_channels**: Defines the minimum number of channels in the convolutional layers inside the generator and discriminator.  **Default**: 64

**add_noise**: If True, adds gaussian noise to the input images when training the discriminator. Helps regularize the discriminator. **Default**: True

**noise_magnitude**: Factor to multiply the gaussian noise with if ```add_noise``` is True. **Default**: 0.1

**custom_G**: None     

**custom_D**: None

**do_plot**: Plots some examples of generated images during training. Recommended to set to False if not running in a notebook, as the plotting will otherwise pause the training procedure. **Default**: True       

**plot_interval**: If ```do_plot``` is True, defines the number of epochs during training between each set of generated images. **Default**: 1

**manual_seed**: 0

**save_gif**: True

**model_folder**: 'models'
