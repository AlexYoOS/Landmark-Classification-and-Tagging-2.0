# Landmark Classification and Social Media Tagging


## _Showcase of the core workload of a supervised learning task in Computer Vision_



This supervised learning task involves training an artificial intelligence (AI) model to learn a mapping between input data and corresponding output labels. This project revolves around teaching the AI model to correctly identify 50 different landmarks based on their features or characteristics.


## Structure and Features of this Showcase

**Data Preprocessing:** This includes resizing images to a uniform size, normalizing pixel values, and augmenting the data with techniques like rotation, flipping, or brightness adjustments to make the model more robust.
**Feature Extraction:** Feature extraction involves converting the images into numerical representations that the AI model can understand. For computational efficiency, transfer learning was used for knowledge transfer and less computing.
**Model Training:** The model is trained using the training and validation dataset. During training, the model learns to map the input images to their corresponding landmark labels. This is done through a process of adjusting model parameters (weights and biases) using optimization algorithms such as gradient descent to minimize the prediction error.
**Hyperparameter Tuning:** Various hyperparameters, like learning rate, batch size, and network architecture, are tuned to optimize the model's performance.
**Validation and Testing:** The model's performance is evaluated on the test set to assess its accuracy and generalization to new, unseen data. Accuracy as the key metric is used to quantify the model's performance.
**Deployment:** The mode of deployment in this showcase is a command-line application, where it can classify landmarks in new images.

>The project outlines the key steps to implement this supervised learning task. Parts like data gathering and labeling, as well as more tailored approaches like "fine-tuning" to the dataset and model deployment in the cloud, have been omitted. This showcase demonstrates the ability to conduct a computer vision problem from idea to solution upon request.


## Tech

This application uses a number of open source projects as tools and dependencies:

- [Python](https://docs.python.org/3/) Core programming language in AI and Data Science
- [Pytorch](https://pytorch.org) - Core artificial intelligence framework
- [Torchvision](https://pytorch.org/vision/stable/index.html) - Computer vision specific sub-library of pytorch
- [Matplotlib](https://matplotlib.org) - Comprehensive library for creating static, animated, and interactive visualizations in Pythoncd 
- [Numpy](https://numpy.org) - The fundamental package for scientific computing with Python
- [PIL](https://pillow.readthedocs.io/en/stable/) -The Python Imaging Library


Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernable landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgement to classify these landmarks would not be feasible.

In this project, you will take the first steps towards addressing this problem by building a CNN-powered app to automatically predict the location of the image based on any landmarks depicted in the image. At the end of this project, your app will accept any user-supplied image as input and suggest the top k most relevant landmarks from 50 possible landmarks from across the world.


## Instructions

#### Setting up locally

1. Open a terminal and clone the repository.
    
2. Create a new conda environment with python 3.7.6:

    ```
        conda create --name landmark_project -y python=3.7.6
        conda activate landmark_project
    ```
        
3. Install the requirements of the project:

    ```
        pip install -r requirements.txt
    ```

4. Install and open Jupyter lab:
	
	```
        pip install jupyterlab
		jupyter lab
	```

### The Notebooks

1. Running the custom build CNN: Open `cnn_from_scratch.ipynb` notebook and follow the instructions there
2. Running the Transfer Learning Implementation: Open `transfer_learning.ipynb` and follow the instructions there
3. Running the APP: Open 'app.py' and follow instructions there (Note: If app is not running in your environment please install jupyter widget extensions:
```
 !jupyter nbextension enable --py widgetsnbextension
```

## Dataset Info

The landmark images are a subset of the Google Landmarks Dataset v2.
