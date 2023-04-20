 <h2> Classification Using CNN Tensorflow </h2>
 
 <p> In order to classify using CNN by Python you need to import the following Packages. </p>
 <code>
 import tensorflow as tf
 from tensorflow import keras
 import pandas as pd
 import matplotlib.pyplot as plt
 from keras.utils import to_categorical
 </code>
 
 <H3> Dataset </H3>
  
 <p> After that we must include our dataset. In this project we have used Fashion MNIST dataset where you can see details in <a href='https://keras.io/api/datasets/fashion_mnist/'> this link </a> </p>
 
 <img src='https://raw.githubusercontent.com/AIAML/Multi_Layer_perceptron_using_Tensorflow/master/fashion-mnist-sprite.png' style='width:800px' /> 
<code> 

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

</code>
<p> Our Train dataset contains 60000 samples so as for to prevent overlearning we have set 10000 for validation set. The following code used for this purpose. </p>

<code>
 train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
 </code>
 
 <p> Next We have to build our model. In this sample we have applied Sequential model of neural network.  </p>
 
 <code> 
 <h3> Model </h3>
 
model = keras.models.Sequential()

</code>
<p> The next step is building our layers. The first layer is formed based on our input. Our Input data is an image which has 28*28 dimenstion. As a consequence our code in python would be:  </p>

<code> 
  model.add(keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(28, 28,1)))

</code>
 
 <p> For selecting your model you have to consider this issue that it takes a while to be proficient in selecting your proper model. In this problem we need to classify images so we have two hidden layer which use 'relu' function. Finally for the last layer we have used softmax function with 10 point which indicates our classes.  </p>
 
 <code>
  model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))

 </code>
 <p>
 In addtion, before fitting you can view a summery of your model.
 </p>
 <code>
  model.summary()
 </code>
 <h3> Fit and Evaluate </h3>
 <p>
 The last step is compiling and fiting our model. Pure and Simple
 </p>
 <code>
  model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,
          batch_size=100,
          epochs=5)
 </code>

<p> 
 For showing every epoch result we can use matplotlib which is available in python and at the beginning we have imported it in our project. Here we have our code using data on <i> history varaible </i> for illustrating.
 </p>
 
 <code>
 
 pd.DataFrame(history.history).plot(figsize=(8, 5))
 plt.grid(True)
 plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
 plt.show()
 </code>
 
 
 
 
 
 
