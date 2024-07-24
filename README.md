# Fruit classification using Siamese Neural Network
This is a team project implemented as a part of a university course of the IT 
specialization in the field of Automation and Robotics, 
in collaboration with the InsERT company. 

The project is Open Source and was undertaken for educational purposes.

Main technologies used: Python, Jupyter Notebook, Tensorflow, Keras, Numpy,  

## Project objectives
The main goal of the project was to create a neural model capable of 
an accurate multiclass classification of different fruits based on their images.
We wanted to adjust our model so as it would perform well in a potential
scenario of being used in self-checkouts in shopping malls. An accurate 
classification of a scanned fruit laying on a checkout could potentially
speed up the buying process and consequently improve users' satisfaction.

We decided that we will provide a user with 5 most accurate predictions made
by model. It would be very risky to show the user only one product since it is 
unlikely that the model will show the correct fruit with a 100% accuracy.

What was also important was to enable an easy addition of new fruits to
the model without the need of repeating the training. That was the reason why we
used the Siamese Neural Network architecture. We implemented both 
contrastive loss and triplet loss approaches. Thanks to this solution, the only
action needed from the staff members, in order to add a new fruit, is to take
5 pictures of the fruit and insert them to the system.

## More about the project
More information about the project is provided in the **Documentation** section.
Note that some of the documents are in polish since they were primarily used
as the final project report for a university course.

In file **SiameseNetwork.pdf** there is a more insightful description
of the decisions made during the development. It is also described what
datasets were used to train the model.

In file **ModelTests.pdf** you can find different training results of our model
and parameters used during subsequent training sessions.

## Results
We managed to achieve a TOP-5 accuracy of **86-87%**. The results are promising,
yet they can be improved during additional training sessions using different 
parameters and datasets.

## Future of the project
There are few possible improvements to be made:
- Improve the TOP-5 accuracy of the model
- Experiment with the TOP-1 or TOP-3 metrics
- Implement an application with an easy-to-use UI which would
  enable users to upload an image and see the results provided by the model.
  The application could also have an option of adding new fruits to the system.