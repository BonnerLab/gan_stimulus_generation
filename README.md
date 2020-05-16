# Generating stimuli

## Optimizing for intended ROI activity
Running the `optimize_for_roi.py` file will generate a set of images
that match the ROI activity of some given targets.

***Note**: Please contact `eric.elmoznino@gmail.com` to obtain the GAN generator and an example encoder trained on the PPA ROI of the Bold5000 dataset.

#### Script arguments
**--save_folder**: A path where the generated images will be saved.\
**--encoder_file**: The name of the pre-trained encoder file to use (must be saved under `saved_models/`)\
**--targets_folder**: A path to the folder containing ROI targets. Each ROI target must be saved as a `.pth` file
containing a 1D tensor. For example, these targets may be ROI voxels predicted by the encoding model for several stimuli,
where each `.pth` file corresponds to the predictions for one stimulus.\
**--n_samples**: The number of images to generate for each target. GAN latent vectors begin as random vectors, so
each generated sample will be different. Generating several samples gives a more robust estimate of the features important
to the ROI.

#### Generating ROI targets for new stimuli
I provide a utility script `make_targets.py` that takes in a folder of images and an encoding model as input and generates
ROI targets such that they can be used for the **--targets_folder** argument. Modify the global variables
in the first few lines of `make_targets.py` and then run the script to generate targets for your own dataset.

#### Example run
I have provided an encoder file for the PPA pre-trained on the Bold5000 dataset (contact `eric.elmoznino@gmail.com` to obtain this trained encoder).
I have also provided ROI targets using this encoder on the COCO dataset.
You can generate stimuli using this encoder and set of targets with the command:
```
python optimize_for_roi.py \
--save_folder coco_ppa_generated --encoder_file \
--encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=PPA.pth \
--targets_folder coco/targets_ppa \
--n_samples 10
```

## Optimizing for AlexNet class
The stimulus-generation method is not restricted to use on encoders; it can work with any differentiable model.
As an example, `optimize_for_class.py` shows how to generate stimuli that maximize the activation of desired AlexNet class labels.
The **--classes** argument must simply be set to a list of ImageNet classes you with to generate stimuli for, and the
**--save_folder** specifies the save location of the generated images.
