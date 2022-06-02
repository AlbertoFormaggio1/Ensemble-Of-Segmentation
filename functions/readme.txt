 ~ stretch_contract_new

Trasformation consists in a stretching.The stretching directions are four: left/up, right/up, left/down, left/up.
Each direction has the same probability of being selected. The stretching is applied to the columns first,
then the empty spaces are interpolated with a weighted nearest neighbour method, and the same
procedure is repeated for the rows.
The stretching is achieved by mapping every row/column of the image to a new position.
The same method is used for the labels.

~ shadows_new

Random shadowing from either the left or right side of the image is introduced in the training data. Shadowing is achieved
by multiplying the intensities in each column of the image by the following function: 
yval = 0.2+((xval./0.5).^(1/2)).*0.8              direction=1
yval = 0.2+(((-xval+1)./0.5).^(1/2)).*0.8       direction=0
The same method is used for the labels.


~ contrast_blur_h_new

The contrast is increased or decreased by mapping the original values to new values according to a mathematical function.
In particular, the following mathematical functions are used:
1) yval = ((xval-1/2).*sqrt(1-k/4))./sqrt(1-((xval-1/2).^2).*k)+0.5

2)yval = 0.5*(xval/0.5)^q               val<1/2
   yval= 1-0.5*((1-xval)/0.5)^q        val>1/2

xval are all possible normalized values, q e k are two parameters that regulate the contrast and that are chosen
randomly from a specific interval.
Finally, a filter is applied to simulate blur due to camera jitter.
The same method is used for the labels.

~ imagesAugmentation_new

To get new images the following operations are applied: change saturation levels, change brightness levels,
change contrast levels, add Gaussian noise, blur. Finally, the image and label are rotated by + - 90 degrees.

~ personalImageAugmentationFunction_new

The image is segmented according to 3 different colors, then divided into 3 images, each containing
one of the 3 colors. The images are combined. Finally, the image and label are rotated by + - 90 degrees.


~ModifiedRandAugument

There are 21 transformations divided into two categories: color and shape. The transformations of the color category are 13: fusion, 
gauss_noise, saturation, contrast, brightness, sharpness, motion, equalize, equalize_yuv, disk_filter, salt & pepper_noise, hue, 
local_contrast. The transformations of the shape are 8: rotation, shearX, shearY, XTranslation, scale, Ytranslation, cutout, flip.
Step 1. For each image and label in the training set, randomly select a transformation from {color}, and implement the transformation
at the preset probability of P.
Step 2. Randomly select a transformation from {shape}, and follow the same procedure in Step 1.


~ricap

The transformation consists of 3 steps.
Step 1.  Randomly select four images k ∈ {1, 2, 3, 4} from the original dataset.
Step 2. Crop the images separately.
Step 3. Patch the cropped images to construct a new image.
The same method is used for the labels.

~imagesTrasformation

Ten artificial images are created for each image in the training set.  The operations performed are:

1)  Width shift → The image is shifted left or right.
2) Height shift→  The image is shifted up or down.
3) Rotation →  Degrees are selected randomly.
4) XShear or YShear → Randomly selected.
5) Vertically or horizontally flip →  Randomly selected.
6) Brightness_1 → A value is added to the actual pixel value for all RGB channels equally.
7) Brightness_1 → A value is added to the actual pixel value for each RGB channels independently.
8) Speckle noise
9) contrast_blur_h_new → Described previously.
10) stretch_contract_new(data) → Described previously.


~occlusion
The images are transformed by replacing rows or columns (randomly) with black lines.The distance between two lines
 is chosen randomly.


~occlusion_2
The images are transformed by replacing some parts of the images with black rectangles. The number of rectangles 
and their size is chosen randomly.

~gridMaskModified

Given an input image, the algorithm randomly removes some pixels of it.
The equation used is ˜x = x × M where x  represents the input image, M ∈ {0, 1} is the binary mask that stores pixels
to be removed, and ˜x is the result produced by the algorithm. The shape of M looks like a grid. 

~ModifiedAttentiveCutMix

The central idea is to create a new artificial image by combining two original images from the training set.
The equation used is: x˜ = B .* x1 + (1 − B) .* x2. 
x1 and x2 are two images of the training set,  ˜x is the result produced by the algorithm, 1 is a binary mask filled with ones, 
B ∈ {0, 1} is a binary mask indicating which pixels in image 1 and 2 will be removed and which ones will remain. 
We first obtain a 7×7 grid map of the first image. We then select the top “N” patches from this 7×7 grid  to cut from the
given image.  The patches are cut from the first image and pasted onto the second image at their respective original locations.

~resizemix
For each image A of the training set, a second image B from the training set is randomly selected. Image B is resized with a random 
dimension and subsequently pasted on top of image A in a random position.Then the function  imagesAugmentation_new described 
previously is applied.


~demo_new
For each image A of the training set, a second image B from the training set is randomly selected. Three methods of color 
normalization of image A versus B are used: 

- Stain Normalisation using RGB Histogram Specification Method
- Stain Normalisation using Reinhard Method
- Stain Normalisation using Macenko's Method

~imagesTrasformation_new

Ten artificial images are created for each image in the training set.  The operations performed are:

1)  Width shift → The image is shifted left or right.
2) Height shift→  The image is shifted up or down.
3) Rotation →  Degrees are selected randomly.
4) XShear or YShear → Randomly selected.
5) Vertically or horizontally flip →  Randomly selected.
6) Brightness_1 → A value is added to the actual pixel value for all RGB channels equally.
7) Brightness_1 → A value is added to the actual pixel value for each RGB channels independently.
8) Speckle noise
9) contrast_blur_h_new → Described previously.
10)shadows_new → Described previously




