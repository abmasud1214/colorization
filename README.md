# colorization
Links to Models and Images:
https://drive.google.com/drive/folders/1k38p4qOigReJTuxQ9jYicBX8Pxy-A04g?usp=sharing
model_epoch1, model_epoch2, and model_epoch3 are the models trained on ImageNet at the relavent epochs.
The Models folder contains the generator models and classifier for the MultiCAN.
INepoch(1 2 5)\_results are the output files from the respective ImageNet models from coloring the test images.
clustered_results are the output files from the MultiCAN from coloring the test images.
256_results are the output files from the model trained on the Caltech256 dataset.
test contains the test images (currently stored as numpy files.)
merged_test_imagenet contains images that have the black and white image (left), output image from imagenet model (middle), and original image (right) from the test set.

Link to website used to evaluate models:
https://thesis-gan-recoloring.herokuapp.com/
Source code:
https://github.com/Jmarkaba/Thesis-Game

Explanation for files:
p2pgan.py contains the code to create the Pix2Pix gan.
trainp2p.py has the appropriate code to load images and train the Pix2Pix GAN. 
preprocessimg.py goes through images in a folder and converts them to the appropriate format expected for training.
modeltest.py loads a model and test folder and generates the output images.
MultiCAN/clusterclassifier.py contains the code for the image classifier on the test set.
MultiCAN/clusteredp2p.py contains the code for the MultiCAN along with the code necessary to train it.
MultiCAN/p2pgan.py is a duplicate of p2pgan.py
MultiCAN/trainclusteredp2p.py creates a MultiCAN model and trains it.
MultiCAN/predictclusteredp2p.py loads a MultiCAN model and predicts images from the test set.
MultiCAN/testclassifiers.py tests the accuracy of the classifier.
