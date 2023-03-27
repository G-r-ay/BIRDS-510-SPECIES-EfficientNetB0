# Bird Species Classification Project

This project is focused on building a machine learning model that can accurately classify bird species based on images. The model was built using PyTorch and the EfficientNet B0 architecture, and achieved a testing accuracy of 0.76.

## Dataset

The dataset used in this project is the 510 Bird Species dataset, which consists of images of 510 different bird species.

## Model

The model used in this project is a convolutional neural network (CNN) based on the EfficientNet B0 architecture. The model architecture consists of several convolutional layers followed by max pooling layers, and a fully connected layer at the end. The model was trained using PyTorch and achieved a testing accuracy of 0.76.

## Repository Contents

The repository contains the following files:

- main.py: A Python script containing the code used to build and train the model.
- model_state_dict.pth: The state dictionary of the trained model.
- README.md: This file.

## Usage

To use the trained model to classify bird species, you can run the main.py script and pass the path of the image file you want to classify as a command line argument. The script will load the state dictionary of the trained model and apply it to the input image, returning the predicted bird species.

Here's an example command to run the script:

```python main.py path/to/image.jpg```


The output will be the predicted bird species, along with the model's confidence level.

## Conclusion

This project demonstrates the use of deep learning techniques for bird species classification based on images. The trained model achieved a testing accuracy of 0.76 and can be used to classify new bird images. The model_state_dict.pth file is included in the repository for easy access to the trained model.
