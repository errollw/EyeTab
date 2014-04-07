# EyeTab

_EyeTab_ is a __3D model-based gaze tracker__ that runs entirely on unmodified commodity tablet computers, taking their limited computational resources and low quality cameras into account.

![Image of EyeTab](https://raw.githubusercontent.com/errollw/EyeTab/master/EyeTab.jpg "Image of EyeTab")

The code is available in three forms:

* `EyeTab` &ndash; A demonstration version which runs on a supplied example video file.
* `EyeTab_SP2` &ndash; An interactive version developed for a _Microsoft Surface Pro 2_. This should also work on other devices with small tweaks.
* `EyeTab_Python` &ndash; A previous iteration of the system written in Python. This was developed rapidly as a prototype so is only included for your curiosity, it is not documented or supported.

The project's webpage can be found [here](http://www.cl.cam.ac.uk/research/rainbow/projects/eyetab/).

A video of the system in action (Python version) can be seen [here](https://www.youtube.com/watch?v=lPcjQdSzKX4).

### Publication

*Erroll Wood and Andreas Bulling. 2014. EyeTab: model-based gaze estimation on unmodified tablet computers. In Proceedings of the Symposium on Eye Tracking Research and Applications (ETRA '14)* [[available at ACM-DL]](http://dl.acm.org/citation.cfm?id=2578185&CFID=433705372&CFTOKEN=17651040)

If you use or extend EyeTab code in full or in part, please cite the paper above.

## System overview

We track gaze by modelling the iris as a 2D ellipse in an image, and _back-projecting_ this to a 3D circle, getting the real-world position and orientation of the iris. We take the normal vector of this to be the gaze direction.

The system has three main components:

1. We first precisely find regions-of-interest for the eyes in an image.
2. Then we robustly fit a 2D ellipse to each _limbus_ &ndash; the boundary between iris and sclera.
3. We finally back-project these to 3D circles, and intersect their normals with the screen for a point-of-gaze.

# Deployment instructions

A rough guide for setting this code up from scratch:

1. Open the solution file `EyeTab.sln` in Visual Studio. I used VS2012.
2. Add dependencies to Visual Studio' _Additional include directories_ field, _Additional library directories_ field, and as _Additional input_ in the linker.
3. Include the `EyeTab` header files in the `EyeTab_SP2` project.
4. Ensure the required `.dll`s can be found on your `PATH` or in Visual Studio's debugging environment.
5. Build and run the solution.

## Dependencies

The system has several dependencies:

* [OpenCV](http://opencv.org/) &ndash; a multi-purpose computer vision library
* [TBB](https://www.threadingbuildingblocks.org/) &ndash; for parellization
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) &ndash; provides vector maths

In addition, the SP2 version of the system depends on [VideoInput](// see: http://www.codeproject.com/Articles/559437/Capturing-video-from-web-camera-on-Windows-and) for providing high-resolution access to the front-facing camera. OpenCV's camera API is broken and does not support this. This library `videoInput.lib` is supplied.

__NOTE:__ The VideoInput library only works when targeting 32-bit, but the rest of the system works fine with 64-bit.
