# ConvNet-Prostate
## Neural Networks for Prostate Cancer Diagnosis

This project is an open source prototype of an application for Prostate Cancer Carcinoma diagnosis.

There are currently two models, one for four different malignancy levels of prostate cancer: Gleason 3 Non Cribiform, Gleason 4 Cribiform, Gleason 4 Non Cribiform, and Normal Prostate.

The other model for simple cancer and non cancer differentiation.


# Usage
---------
In order to start training, place data in the respective folders for VALIDATION and General-Data.
The following command will start the training process:

First get to the directory where the code is stored:
```
$Directory python main.py
```
Once the training sequence is completed: there is an option of 

```
$Directory python retrain.py
```
or
```
$Directory python eval.py
```

ConvNet Prostate stores training image data in a very specific way. See the the CIFAR10 Data format for more information.
If you wish to change the image size, go into Input.py and change:

```python

  label_bytes = 1
  result.height = (your image size)
  result.width = (your image size)
  result.depth = 3
```
The depth can also be altered if you are using grayscale, though thats strongly recommended against.

# Results
---------

- **For four malignancy levels, the accuracy is 51% compared with a random guess of 25%.**
- **For non cancer and cancer diagnosis, the accuracy is 90%.**

Validation was done on a total of 64 samples due to limited data.

ConvNet-Prostate is open sourced for transparency and academia; however, the data used is propietary.

