---
layout: ../../layouts/MarkdownLayout.astro
title: 'Digit recognizer from scratch'
description: 'Python based neural network library built from scratch used to create a digit recognizer trained on the MNIST dataset'
tags: ["Python", "Machine learning"]
---
# Introduction 

This project is a digit recognizer built entirely from scratch in Python, without relying on high-level machine learning frameworks like TensorFlow or PyTorch. The goal of this project was not just to achieve good accuracy on the MNIST dataset, but to deeply understand how neural networks work under the hood: forward propagation, backpropagation, gradient descent, and the practical challenges that come with training models.

To accomplish this I built a small neural network library that handles layers, activations, loss functions, and training loops, and then used it to train a classifier capable of recognizing handwritten digits from the MNIST dataset. 
## Motivation
This project had been a personal goal of mine even before I learned how to program. With a background in university‑level calculus, I was particularly drawn to the mathematical foundations of neural networks. My initial motivation came from the [3Blue1Brown YouTube series on neural networks](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=GRKGyh973JjkTuv4), which clearly illustrated how relatively simple mathematical operations can give rise to complex pattern recognition and got me interested in the world of machine learning.

Working on this project deepened my understanding of neural networks (including backpropagation for general layers, different loss functions, and convolutional architectures), as well as broader machine‑learning concepts such as data normalization and train–test splits. It also strengthened my general programming skills, especially in object‑oriented design and the use of patterns such as composition.

Although the original goal was simply to build a digit recognizer, the project naturally evolved into a more general neural‑network library. Abstracting the code in this way made it possible to experiment with different architectures and concepts beyond classification, such as variational autoencoders.
## Project Overview
Using the library, I built several models with different architectures that achieved high accuracy. All models take as input a 28×28 grayscale image and output a probability distribution over the digits 0–9, representing the model’s confidence that each digit is present in the image.
The models are trained on the MNIST dataset and I created a simple terminal user interface to try the model on user created data.
## Neural Network Implementation
### Structures
The models in this project are instances of the NeuralNetwork class, which contains an array of Layer objects. 
Layer objects must implement the following methods.
```python
forward(self, Input)
backward(self, grad_output, dt)
```
These methods are also defined in the NeuralNetwork class based on the implementation in its layers and they are involved in the response of the network to a new image and in the training loop respectively.
```python
def forward(self, Input):
    for layer in self.layers:
        output = layer.forward(Input)
        Input = output
    return output

def backward(self, grad_output, dt):
    for layer in reversed(self.layers):
        grad_output = layer.backward(grad_output, dt)
    return grad_output
```
### Forward Propagation
The forward propagation is the process of getting a response from the model to a particular input, the input is handled by the first layer, it gets processed by the layer's parameters and it returns an output, this output is then used by the next layer in a loop until the last layer which returns the output of the model.
```python
def forward(self, Input):
    for layer in self.layers:
        output = layer.forward(Input)
        Input = output
    return output
```
### Loss Function
The training method requires a loss function to minimize, this usually comes in the form of a function $L(Y, Y_{true})$ with a computable derivative with respect to its inputs $\nabla_Y L (Y, Y_{true})$. 
```python
def costo(output, output_correcto):
def dev_costo(output, output_correcto):
```
For the digit recognizer the most suited loss is the cross-entropy because it computes a relevant difference metric between two probability distributions, the predicted and the true one(which is 100% for the correct digit and 0% for the rest):
```python
def cross_entropy(output, output_correcto):
    return -np.sum(output_correcto * np.log(output))
def dev_cross_entropy(output, output_correcto):
    return -output_correcto / output
```
Other loss functions were defined aswell for experimentation.
### Backpropagation
The gradient of the loss function with respect to the model's output $\nabla_Y L$ represents the optimal infinitesimal change in the output of the model to an input for improving performance, we cant modify this output directly, but we can change the model's parameters to change the output. For this, we need a formula for how to change a layer's parameters to minimize the loss,  this is done by gradient descent $\theta \leftarrow \theta - \nabla_\theta L \cdot dt$, this requires an explicit formula for $\nabla_\theta L$, which can be obtained by the chain rule, since a layer is a function $y = f(x, \theta)$, if we know $\nabla_y L$, then $\nabla_\theta L$ is $\nabla_y L \cdot \nabla_\theta y$ by the chain rule, for the last layer $\nabla_y L$ is the gradient of the loss $\nabla_Y L (Y, Y_{true})$, for a general layer its output is the input of the next layer, so it's $\nabla_y L$ is the next layer's $\nabla_x L$, so every layer should also calculate $\nabla_x L$, which is $\nabla_y L \cdot \nabla_x y$ by the chain rule, and pass it to the layer before it. Both $\nabla_\theta L$ and $\nabla_x L$ have to be computable within the layer's scope (using only $x$ , $y$ and $\nabla_y L$).       

So the backpropagation step for a layer is then:
- Get $\nabla_y L$ as an input to the step.
- Compute the gradient with respect to the layer's parameters $\nabla_\theta L=\nabla_y L \cdot \nabla_\theta y$
- Modify the parameters according to gradient descent $\theta \leftarrow \theta - \nabla_\theta L \cdot dt$
- Compute the gradient with respect to the layer's input $\nabla_x L=\nabla_y L \cdot \nabla_x y$
- Return as an output the gradient with respect to its input $\nabla_x L$

An example of a layer's backpropagation step in the dense (also known as linear) layer:
```python
def backward(self, grad_output, dt):
    grad_pesos = grad_output @ self.input.T
    self.pesos -= grad_pesos * dt
    grad_input = self.pesos.T @ grad_output
    return grad_input
```
And for a full network the backpropagation process works as follows:
```python
def backward(self, grad_output, dt):
    for layer in reversed(self.layers):
        grad_input = layer.backward(grad_output, dt)
        grad_output = grad_input
    return grad_input
```
### Architecture
The best performing model has the following shape:
```python
red = NeuralNetwork(
    input_shape,
    [
        ConvolucionalNoBias(kernel_size=3, kernels=2),
        Leaky_Relu(),
        Max_Pooling(),
        ConvolucionalNoBias(kernel_size=3, kernels=5),
        Leaky_Relu(),
        Max_Pooling(),
        Flatten(),
        Densa(output_neurons=10),
        Softmax()
    ],
)

```
```
shape:
    ConvolucionalNoBias
        (1, 28, 28)->(2, 26, 26)
    Leaky_Relu
        (2, 26, 26)->(2, 26, 26)
    Max_Pooling
        (2, 26, 26)->(2, 13, 13)
    ConvolucionalNoBias
        (2, 13, 13)->(5, 11, 11)
    Leaky_Relu
        (5, 11, 11)->(5, 11, 11)
    Max_Pooling
        (5, 11, 11)->(5, 5, 5)
    ConvolucionalNoBias
        (5, 5, 5)->(10, 3, 3)
    Leacky_Relu
        (10, 3, 3)->(10, 3, 3)
    Max_Pooling
        (10, 3, 3)->(10, 1, 1)
    Flatten
        (10, 1, 1)->(10, 1)
    Softmax
        (10, 1)->(10, 1)
```
Whereas the simplest model consists of a single Dense layer and a Softmax layer to make the output a valid probability distribution.
```python
red = NeuralNetwork(
    input_shape,
    [
        Flatten(),
        Densa(output_neurons=10),
        Softmax()
    ],
)
```
```
shape:
    Flatten
        (1, 28, 28)->(784, 1)
    Densa
        (784, 1)->(10, 1)
    Softmax
        (10, 1)->(10, 1)
```
## User interface
The idea of the project was not just to create a model that would perform well on the MNIST dataset, but one that could also perform well on images created by the user, for this I built a small CLI where you can drag the image file to the terminal and paste its filepath and let the model guess the number. The images can be easily created in MS paint and then processed to have the correct size within the program. 
## Dataset
### MNIST
MNIST is a standard benchmark dataset of 28×28 grayscale images of handwritten digits, commonly used to evaluate image‑classification models.
### MS paint data
When trained exclusively on MNIST, the model performs poorly on user‑generated images. This is likely because MNIST digits are carefully preprocessed and centered, whereas user‑drawn images differ in style and preprocessing. To address this, the CLI allows users to add their own images to a custom dataset, enabling the model to adapt better to non‑MNIST inputs.


## Training
Training is performed iteratively. In each iteration, the model processes the full dataset, predicting an output for each image and computing the loss with respect to the correct label. The gradient of the loss is then backpropagated through the network to update its parameters.

This approach corresponds to stochastic gradient descent, where parameter updates are computed using individual data points rather than the entire dataset at once. In practice, this method converges faster and often generalizes better than full‑batch gradient descent.

```python
def iteracion(red, datos, loss, dt):
    error = 0
    costo, dev_costo = loss
    for dato in datos:
        Input = dato["input"]
        output_correcto = dato["output"]
        output = red.forward(Input)
        error += costo(output, output_correcto)
        grad_error = dev_costo(output, output_correcto)
        red.backward(grad_error, dt)
    return error / len(datos)

def entrenar(
    red,
    datos,
    num_iteraciones,
    loss,
    dt
):
    for i in range(num_iteraciones):
        error = iteracion(red, datos, loss, dt)
        print(i, error")
```
Models are saved as pickle files and can be reloaded either for inference on user‑provided images or for further training with different hyperparameters. Simpler architectures tend to converge quickly with larger learning rates, while deeper models require smaller learning rates and more iterations but achieve better overall performance.
## Results

<figure>
    <video src="/digit-recognizer/Demonstration.mp4" controls width="100%">
      Your browser does not support the video tag.
    </video>
    <figcaption> Example of the MS paint testing console workflow</figcaption>
</figure>

### Best model
- MNIST test set: 92.47%
- MS paint test set: 83.33333333333333 (10/12)
### Simplest model
- MNIST test set: 82.96%
- MS paint test set: 75% (9/12)

### The weights of the model

The simplest model is a single biasless-linear layer followed by a softmax to create probabilities, a linear layer is just a matrix of learned weights multiplied by the flattened vector version of the imput image. 

The result of matrix product with a vector can be thought of as the list of the dot products of each of the rows of the matrix and the vector.

The linear layers has to take the input vector of size $28 \cdot 28$ and return a vector of size $10$, one component for each digit, for this, the matrix has to have `shape=(10, 28*28)`

So this matrix can be thought of 10 rows of size $28 \cdot 28$, just like the input, and the dot product of each of those with the vector is the sum of the product of each of it components with the components of that row, inspecting those rows as if they were images can shed some light into what the model is aactually learning.

$$
W \cdot \vec v = \begin{pmatrix}
    \vec w_0 \\
    \vec w_1 \\
    \vec w_2 \\
    \vec w_3 \\
    \vec w_4 \\
    \vec w_5 \\
    \vec w_6 \\
    \vec w_7 \\
    \vec w_8 \\
    \vec w_9 \\
\end{pmatrix} \cdot \vec v = \begin{pmatrix}
    \vec w_0 \cdot \vec v\\
    \vec w_1 \cdot \vec v\\
    \vec w_2 \cdot \vec v\\
    \vec w_3 \cdot \vec v\\
    \vec w_4 \cdot \vec v\\
    \vec w_5 \cdot \vec v\\
    \vec w_6 \cdot \vec v\\
    \vec w_7 \cdot \vec v\\
    \vec w_8 \cdot \vec v\\
    \vec w_9 \cdot \vec v\\
\end{pmatrix} = \begin{pmatrix}
    a_0 \\
    a_1 \\
    a_2 \\
    a_3 \\
    a_4 \\
    a_5 \\
    a_6 \\
    a_7 \\
    a_8 \\
    a_9 \\
\end{pmatrix} 
$$

<figure>
    <image src="/digit-recognizer/pesos_column.png" width="20%">
    </image>
    <figcaption> Weights of a trained linear model plotted as a list of images </figcaption>
</figure>

The effect of the weights is clearest in the weights for the 0 digit, we can clearly see high positive values forming a circle and negative values inside the circle, so an image with pixels forming a circle would highly match the weight, whereas an image with its center painted would have a much smaller dot product.


## Limitations

A few of the issues this implementation doesnt handle well are:
- Uncentered digits. The MNIST dataset has its images centered from its preprocessing, this is a step that I didnt recreate and it can cause the model to underperform when the digits arent centered.
- Similar digits: The models can often confuse certain digits with similar shapes like 3's and 8's.
- Compute time: Since I focused this project on my learning of machine learning and programming in general, the implementations of almost all mathematical operations are done by me in python which slows down training time significantly.  

## Repository
Link to the source code and instructions on how to run it.
The source code for this project can be found in the following [github repo](https://github.com/juangriffin121/Digit_recognizer).

