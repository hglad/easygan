# About
An attempt at a user-friendly PyTorch implementation of a Generative Adverserial Network (GAN). The user only needs to supply the data (images) in order to train their own GAN, and parameters such as learning rates and optimizer properties can be tweaked by the user. The generator and discriminator are currently only built for handling 128x128 RGB images.

Most of the testing of this package has been done in a Jupyter Notebook environment, so I highly recommend using this package in an equivalent environment.

# Installation
Clone this repository, navigate to the root folder containing ```setup.py``` and run ```python -m pip install .```

# Usage
#### Train with default parameters
With ```imgs``` provided as a list or NumPy array of images, the following code will create a class instance and train a GAN using the default parameters:
```python
import easygan as eg

imgs_tn = eg.preprocess(imgs)      # normalize pixel values and create tensor
mygan = eg.GAN()          
mygan.train_gan(imgs_tn)       
```
By default, the models (generator and discriminator) will be saved to a folder inside ```models``` in your working directory. Some results will also be generated and saved to the folder ```results```, also placed in your working directory.

Training can be resumed by running ```train_gan``` again. Note that this will save a new set of models rather than overwriting the previous models.

#### Train with other parameters
Parameters can be supplied to the ```train_gan``` function (a list of tweakable parameters is provided in the final section of this Readme), providing some control over the training procedure and more:
```python    
mygan.train_gan(imgs_tn, batch_size=16, epochs=50, DiffAugment=True)       
```

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

**augment_policy**: 'color,translation'

**z_size**: 100    

**base_channels**: 64

**add_noise**: True

**noise_magnitude**: 0.1

**custom_G**: None     

**custom_D**: None

**do_plot**: True       

**plot_interval**: 1

**manual_seed**: 0

**save_gif**: True

**model_folder**: 'models'
