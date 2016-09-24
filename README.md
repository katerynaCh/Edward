###EDWARD
*EDWARD is aimed to allow user running convolutional neural network for image classification and customize it according to the data used. This allows to receive more accurate and justified results in medical imaging analysis.*

####File list
--------------------------------------------------------------------------------------------------------------------------
Classifiers:
* **logistic_sgd.py**		Multi-class logistic regression
* **mlp.py**			Multilayer perceptron
* **convolutional_mlp.py**	Convolutional Neural Network

Interface:		
* **tryconsole.py**		Starts calculations on the console
* **testdesign.py** 	Graphical interface
* **executework.py**		Execution code

####Requirements & Guide
--------------------------------------------------------------------------------------------------------------------------


**Step 1**<br />
Run **executework** to launch interface

**Step 2**<br />
Press "Create!"

**Step 3**<br />
Specify the parameters, searching for the best combination to fit your data

**Step 4**<br />
In "Import Data" and specify the path to your *.zip* file
>**Note:**
*EDWARD* allows you to create classifiers with unlimited amount of classes in the most simple way. Just create one folder per class and name it numericalluy (i.e. 0, 1, 2, etc.), then place all the *.jpg* images from each class to corresponding folder. After converting the directory to *.zip* file, your dataset is ready. Just select the path to the file in *EDWARD* and press "Submit"
>

**Step 5**<br />
Press "Submit"

**Step 6**<br />
After the appearance of the console output window, press "Start" and wait for the model to train. As it is done, you will be able to scroll the window and analyse the performance

>**Important Note:**
Currently, load\_data function of the model is still under development. So in order to run the model, one has to specify the path manually. This can be done by setting **dataset=" "** variable in *convolutional\_mlp* script to the correct path to the unzipped folder (if you want to use our demo-data, provide a path to the 'testing' folder that you download from the "data" folder)
>

**Packages:** theano, numpy, scipy, pillow, six<br />
`pip install theano`<br />
`pip install numpy`<br />
`pip install scipy`<br />
`pip install pillow`<br />
`pip install six`<br />
**Python version:** at least 3.4.X<br />
**Data Format:** zip. file<br />
**IMG format:** .jpg<br />


####Parameters
-------------------------------------------------------------------------------------------------------------------------
Basic settings:
* **Amount of layers:** № of convolutional layers.
* **Batch size:** № of training examples in a forward-back pass.
* **Amount of epochs:** № of forward-backward pass of all the training examples.
* **Size of pooling:** Pooling partitions the input image into a set of non-overlapping rectangles.

Advanced:
* **GPU:** For higher performance GPU might be used.
* **Size of filter:** Size of a matrix looking for a particular feature (depends of image size).
* **Learning rate:** Control of weights and biases.

>**Note**:<br />
We restric parameters to optimal numbers in order to minimize human error.
><br />

*Sincerely,*<br />
*Stayin' Alive Team.*
