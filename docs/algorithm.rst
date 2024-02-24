Algorithm
=========

We first preprocesses the data by splitting it up into discrete pixels of 1 micrometer.

We then assign pixels as being either background, border, or nucelus based on
the distance of the pixel to the nearest nucelus or density of transcripts.

For all pixels inside the nucleus, we calculate the angle of the vector from the pixel to the
centroid of the nucleus.

We use a EM approach to estimate the expression profiles of cell types in the data, and then choose
a number of cell types based on AIC/BIC. We assign each nucleus to a cell type based on the transcripts it contains.

Next we use a UNet with number of channels equal to the number of genes, and number of output classes
equal to the number of celltypes. We also add to this two special output classes: one for background/foreground discrimination,
and another that represents the angle of the vector from the pixel to the centroid of the nucleus
it belongs to.

We train the UNet using pixels which are contained in the nucleus. This allows us to train the UNet to classify
angle to the centroid of the nucleus, and the cell type of pixels. We train the foreground/background classification
using the foreground/background labels we assigned to the pixels based on our nucleus/transcript distance thresholds.

Once the UNet is trained, we use it to predict the nucleus centroid angle of all pixels in the image.
We then use a greedy expansion algorithm to expand each nucleus, adding pixels to that cell if the angle
of that pixel would intersect the nucleus body given a certain vector magnitude.
