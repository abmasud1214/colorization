# colorization
Links to Models and Images: <br />
https://drive.google.com/drive/folders/1k38p4qOigReJTuxQ9jYicBX8Pxy-A04g?usp=sharing <br />
model_epoch1, model_epoch2, and model_epoch3 are the models trained on ImageNet at the relavent epochs. <br />
The Models folder contains the generator models and classifier for the MultiCAN. <br />
INepoch(1 2 5)\_results are the output files from the respective ImageNet models from coloring the test images. <br />
clustered_results are the output files from the MultiCAN from coloring the test images. <br />
256_results are the output files from the model trained on the Caltech256 dataset. <br />
test contains the test images (currently stored as numpy files.) <br />
merged_test_imagenet contains images that have the black and white image (left), output image from imagenet model (middle), and original image (right) from the test set. <br />
<br />
Link to website used to evaluate models: <br />
https://thesis-gan-recoloring.herokuapp.com/ <br />
Source code: <br />
https://github.com/Jmarkaba/Thesis-Game <br />
<br /> 
Explanation for files: <br />
p2pgan.py contains the code to create the Pix2Pix gan. <br />
trainp2p.py has the appropriate code to load images and train the Pix2Pix GAN.  <br />
preprocessimg.py goes through images in a folder and converts them to the appropriate format expected for training. <br />
modeltest.py loads a model and test folder and generates the output images. <br />
MultiCAN/clusterclassifier.py contains the code for the image classifier on the test set. <br />
MultiCAN/clusteredp2p.py contains the code for the MultiCAN along with the code necessary to train it. <br />
MultiCAN/p2pgan.py is a duplicate of p2pgan.py <br />
MultiCAN/trainclusteredp2p.py creates a MultiCAN model and trains it. <br />
MultiCAN/predictclusteredp2p.py loads a MultiCAN model and predicts images from the test set. <br />
MultiCAN/testclassifiers.py tests the accuracy of the classifier. <br />
