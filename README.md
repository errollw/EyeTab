# EyeTab

EyeTab is a __3D model-based gaze tracker__ that runs entirely on unmodified commodity tablet computers, taking their limited computational and sensor resources into account.

The code is available in two forms:

* A _basic_ version which runs on a supplied example video file. This should give you an idea of the system's operation.
* An _extended_ version for real-time operation on a _Microsoft Surface Pro 2_. Unfortunately camera APIs are non-trivial, so this is not guaranteed to work on other tablet devices.

## System overview

# Deployment instructions

EyeTab is written in Visual C++, a small extension of C++ which includes ... It should also work fine as generic C++ with minor replacements of VC++ code.

I chose not to include Visual Studio _solution_ and _project_ files as these will vary depending on how your dependency installations are laid out.

A very rough guide for setting this code up from scratch:

1. Make a new empty console application in Visual Studio.
2. Correctly include, link, and input the system dependencies.
3. Copy the source, header and resource files into your empty project.

## Dependencies

The system has several dependencies:

* [OpenCV](http://jquery.com/) - a multi-purpose computer vision library
* [TBB](http://jquery.com/) - for multi-threading
* [Eigen](http://jquery.com/) - provides simple vector maths

In addition, the SP2 version of the system depends on [VideoInput](http://jquery.com/) for providing high-resolution access to the front-facing camera. OpenCV's camera API is broken and does not support this.
