Download Link: https://assignmentchef.com/product/solved-ceng499-introduction-to-machine-learning-homework-1-artificial-neural-networks
<br>
<h1>1           Introduction</h1>

In this assignment, you will have the chance to get hands-on experience with artificial neural networks. The assignment consists of three tasks. Firstly, you will implement a multilayer perceptron (MLP) from scratch by using NumPy library. Secondly, you will optimize the hyperparameters of a neural network. Lastly, you will build a convolutional neural network (CNN) with PyTorch framework.

<h1>2           Multilayer Perceptron (30 pts)</h1>

In this section, you will implement a two layer (one hidden layer) MLP by using NumPy library, a general-purpose array-processing package. No other library or framework is allowed. After building and training the model, you will test it with the dataset we provided. The program will read three command line arguments: the path of the train set, the path of the test set, and the number of epochs respectively. An epoch is one complete pass through the dataset. As the output, the program will print the test accuracy to the screen and terminate. Both sets can be downloaded from ODTUClass.

<ul>

 <li>Write your code in a file called <strong>py</strong>. When grading, we will run it with different datasets and number of epochs.</li>

 <li>The following code can be used for reading the train/test set.</li>

</ul>

<table width="434">

 <tbody>

  <tr>

   <td width="434">import pickle, gzip with gzip.open(dataset_path) as f:data, labels = pickle.load(f, encoding=’latin1’)</td>

  </tr>

 </tbody>

</table>

<ul>

 <li>While the number of nodes at the input layer (<em>n<sub>i</sub></em>) is the number of features, at the output layer (<em>n<sub>o</sub></em>) there is only one node (binary classification task).</li>

</ul>

The number of nodes at hidden layer (<em>n<sub>h</sub></em>) will be  for this task.

<ul>

 <li>The activation function for the output layer is sigmoid function.</li>

</ul>

(1)

For the hidden layer, the activation function is the following piecewise function.

(2)

<ul>

 <li><em>L</em><sub>2 </sub>loss function will be used to train the classifier.</li>

</ul>

(3)

where <em>N </em>is the number of training examples, <em>y </em>is the true label, and ˆ<em>y </em>is the output of the model, the predicted label.

<ul>

 <li>Initialize the weights with small random numbers and the biases with zero. Use the following code and replace … with dimension values of the layers.</li>

</ul>

weight = np.random.randn(…) * 0.01 bias = np.zeros((…))

<ul>

 <li>To get reproducible and consistent results, you need to fix the randomness in your code. Before writing anything else, prepend the following code so that NumPy can always produce the same random number sequence.</li>

</ul>

import numpy as np np.random.seed(1234)

<h1>3           Hyperparameter Tuning (50 pts)</h1>

For the model in the previous section, we specified some parameters such as the number of layers, the number of nodes at a layer, and the type of an activation function. These parameters are called <strong>hyperparameters</strong>. Unlike parameters, they are not optimized via training process. Hyperparameters define the structure of a model, and how the model is trained. In this section, you are going to implement models with different hyperparameter configurations and report the results.

<ul>

 <li>Write your code in a file called <strong>py</strong>. When grading, we run your program with the hyperparameter configurations you tried. You will write the configurations and the test accuracies you achieved in a report named <strong>report.pdf</strong>.</li>

 <li>You will employ PyTorch deep learning framework. No other library or framework is allowed for building the models.</li>

 <li>You are expected to use fully-connected layers in your network.</li>

 <li>To get reproducible and consistent results, you need to fix the randomness in your code. Before writing anything else, prepend the following code so that PyTorch can always produce the same random number sequence.</li>

</ul>

import torch torch.manual_seed(1234)

<ul>

 <li>For this section, CIFAR-10 dataset is selected. The dataset consists of 32 × 32 pixel RGB images, each of which belongs to one of 10 categories. Use the following code to download the dataset. You can use any ratio for the validation set.</li>

</ul>

import torch import torchvision.transforms as transforms from torchvision.datasets import CIFAR10

ratio = 0.1 transform = transforms.Compose(

[transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = CIFAR10(root=”task2_data”, train=True, transform=transform, download=True)

test_set = CIFAR10(root=”task2_data”, train=False, transform=transform, download=True)

train_set, val_set = torch.utils.data.random_split(train_set,

[int((1 – ratio) * len(train_set)), int(ratio * len(train_set))])

<ul>

 <li>The best configuration should achieve at least 50% accuracy in the test set.</li>

 <li>You can choose any optimizer. However, the loss function that is optimized must be <strong>cross entropy</strong>. Be aware that PyTorch’s cross entropy implementation does not need explicit <strong>softmax </strong>activation function.</li>

 <li>You are going to tune the following hyperparameters. You can also experiment on additional hyperparameters.

  <ol>

   <li>Try different number of layers. You should at least try 1, 2, and 3 layer networks. The number of neurons at each layer is not important as long as you meet the minimum test set accuracy.</li>

   <li>Try different activation functions for each layer. You should try different combinations of at least 3 types of activation functions. Possible choices can be <strong>Leaky ReLU</strong>, <strong>GELU</strong>, and <strong>SELU</strong>.</li>

   <li>Try different learning rates. You should try at least 5 different values. Possible choices can be 0.1, 0.03, 0.01, 0.003, and 0.001.</li>

  </ol></li>

</ul>

For each hyperparameter configuration, state the test result accuracy in your report. At minimum, you experiment 5 + 15 + 45 different configurations. It is up to you how you present the results (for example, in the form of a table).

<ul>

 <li>Discuss how you decided the best hyperparameters. What method did you use?</li>

 <li>For each k-layer network, select the best performing hyperparameter configuration and draw its training and validation losses on the same graph and comment on it. What countermeasure did you take against overfitting? How may one understand when a network starts to overfit during training? What method did you use to decide where to end the training procedure?</li>

 <li>Discuss whether accuracy is a suitable metric to measure the performance of a network for this specific dataset. Explain your reasons. (You don’t have to use another metric in your experiments if you can achieve the minimum accuracy required for this homework. Just discuss how using another metric might have changed hyperparameter optimization results.)</li>

</ul>

<h1>4           Convolutional Neural Networks (20 pts)</h1>

In this section, you are going to write a convolutional neural network (CNN) model that works with arbitrary input image sizes by using PyTorch. The dataset that you are going to work on is MNIST, which consists of 28 × 28 grayscale images, each of which belongs to one of 10 categories. However, we apply random croppings and paddings to the images, thereby making the size of the input image to the model arbitrary. In <strong>task3.py</strong>, we have created a skeleton for the model. You are free to design the architecture of your CNN as you like, however, it should have at least two convolutional layers.

<h1>5           Specifications</h1>

<ul>

 <li>Falsifying results, changing the composition of training and test data are strictly forbidden, and you will receive 0 if this is the case. Your programs will be examined to see if you have actually reached the results and if it is working correctly.</li>

 <li>Using any piece of code that is not your own is strictly forbidden and constitutes as cheating. This includes friends, previous homeworks, or the internet. The violators will be punished according to the department regulations.</li>

 <li>Follow the course page on ODTUClass for any updates and clarifications. Please ask your questions on the discussion section of ODTUClass instead of e-mailing.</li>

 <li>You have total of 3 late days for <strong>all </strong>your homeworks. For each day you have submitted late, you will lose 10 points. The homeworks you submit late after your total late days have exceeded 3 will not be graded.</li>

</ul>


