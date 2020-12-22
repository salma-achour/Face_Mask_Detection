# Face Mask Detection

This project aims to detect weather a person is wearing a face mask or not to do so 2 deep learning models were used:
* The pre-trained model `MobileNetV2` was used to detect the existance or not of a mask.
* The pre-trained model `FaceNet` or `GoogleNet` was used for the face detection part.

# The Dataset
The dataset used in a kaggle dataset that can be found [here](https://www.kaggle.com/omkargurav/face-mask-dataset) or in the `Data` folder of this repository.

The dataset is composed of 2 folders:
* `with_mask` : this folder contains 3725 images of people wearing medical masks.

<p float="left">
  <img src="Data/with_mask/0_0_0 copy 11.jpg" width="100" />
  <img src="Data/with_mask/0_0_0 copy 15.jpg" width="100" /> 
  <img src="Data/with_mask/0_0_0 copy 20.jpg" width="100" />
   <img src="Data/with_mask/0_0_0 copy 85.jpg" width="100" />
  <img src="Data/with_mask/0_0_0 copy 92.jpg" width="100" /> 
  <img src="Data/with_mask/0_0_0 copy 2.jpg" width="100" />
</p>

* `without_mask` : this folder contains 3828 images of people not wearing madical masks.

<p float="left">
  <img src="Data/without_mask/0_0_aidai_0014.jpg" width="100" />
  <img src="Data/without_mask/0_0_caizhuoyan_0014.jpg" width="100" /> 
  <img src="Data/without_mask/0_0_chenglong_0070.jpg" width="100" />
  <img src="Data/without_mask/0_0_baobeier_0098.jpg" width="100" />
  <img src="Data/without_mask/0_0_benxi_0129.jpg" width="100" /> 
  <img src="Data/without_mask/0_0_caiyilin_0050.jpg" width="100" />
</p>

![Alt Text](result.gif)
