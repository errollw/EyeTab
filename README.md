# EyeTab

_EyeTab_ is a __3D model-based gaze tracker__ that runs entirely on unmodified commodity tablet computers, taking their limited computational resources and low quality cameras into account.

![Image of EyeTab](https://raw.githubusercontent.com/errollw/EyeTab/master/EyeTab.jpg "Image of EyeTab")

The code is available in two forms:

* `EyeTab` &ndash; A demonstration version which runs on a supplied example video file.
* `EyeTab_SP2` &ndash; An interactive version developed for a _Microsoft Surface Pro 2_. This should also work on other devices with small tweaks.

The project's main webpage can be found [here](http://www.cl.cam.ac.uk/research/rainbow/projects/eyetab/).

## System overview

We track gaze by modelling the iris as a 2D ellipse in an image, and _back-projecting_ this to a 3D circle, getting the real-world position and orientation of the iris. We take the normal vector of this to be the gaze direction.

The system has three main components:

1. We first precisely find regions-of-interest for the eyes in an image.
2. Then we robustly fit a 2D ellipse to each _limbus_ &ndash; the boundary between iris and sclera.
3. We finally back-project these to 3D circles, and intersect the circle normals with the screen for a point-of-gaze.

# Deployment instructions

A rough guide for setting this code up from scratch:

1. Open the solution file `EyeTab.sln` in Visual Studio. I used VS2012.
2. Add dependencies to Visual Studio' _Additional include directories_ field, _Additional library directories_ field, and as _Additional input_ in the linker.
3. Ensure the required `.dll`s can be found on your `PATH` or in Visual Studio's debugging environment.
4. Build and run the solution.

## Dependencies

The system has several dependencies:

* [OpenCV](http://opencv.org/) &ndash; a multi-purpose computer vision library
* [TBB](https://www.threadingbuildingblocks.org/) &ndash; for parellization
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) &ndash; provides vector maths

In addition, the SP2 version of the system depends on [VideoInput](// see: http://www.codeproject.com/Articles/559437/Capturing-video-from-web-camera-on-Windows-and) for providing high-resolution access to the front-facing camera. OpenCV's camera API is broken and does not support this. This library `videoInput.lib` is supplied.
