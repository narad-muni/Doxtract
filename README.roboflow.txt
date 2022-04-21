
sasa - v1 pre
==============================

This dataset was exported via roboflow.ai on March 11, 2022 at 6:52 AM GMT

It includes 948 images.
Sa are annotated in COCO format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random Gaussian blur of between 0 and 4.5 pixels
* Salt and pepper noise was applied to 5 percent of pixels


