# SEMANTIC SEGMENTATION OF CARCASSES USING DEEP LEARNING

For source code refer to .py files. 
For a light demonstration refer to the notebook (Demonstration.ipynb)

## Help

For Main_Training.py the following argument can be used when making the command call:
```
Training script for segmentation models

options:
  -h, --help            show this help message and exit
  -image IMAGE_DIR_NAME
                        Directory containing images
  -label LABEL_DIR_NAME
                        Directory containing image labels
  -model MODEL_PICK     Chose a model type from these; ['fcn', 'unet', 'unet_pretrained', 'dlv3', 'dlv3p']
  -l LOSS               Chose a loss type from these; ['FL', 'TL', 'CE']
  -ne N_EPOCHS          Number of epochs

```

Local_test.py
```
options:
  -h, --help            show this help message and exit
  -image IMAGE_DIR_NAME
                        Directory containing images
  -label LABEL_DIR_NAME
                        Directory containing image labels
  -model MODEL_PICK     Chose a model type from these; ['fcn', 'unet', 'unet_pretrained', 'dlv3', 'dlv3p']
  -model_path MODEL_PATH
                        specify path to model
  -n NUM_IMAGES         Number of test images
  -suffix NAME_SUFFIX   Append given string to output image name
  -dice DICE            Test all test images with Dice scores. Default: False
  -pred_img PRED_IMG    If set to true the function will make an image of the predictions and save to a file. Default: True

```

## Caveats
Our project supervisor does not want us to include images. But a few samples can be seen in Demonstration.ipynb. Similarly, we have not included the final weights as these are valuable to the company and should not be public.
