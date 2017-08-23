# surgical video dataset
This video dataset contains rectified stereo images collected in robotic surgery. The images have been resized to 384x192. Please see our related [source code project](https://github.com/magnumye/disp_est_net) for example usage.

If you use the data, please cite following:

```
Ye, M., Johns, E., Handa, A., Zhang, L., Pratt, P. and Yang, G.Z. 
Self-Supervised Siamese Learning on Stereo Image Pairs for Depth 
Estimation in Robotic Surgery. Hamlyn Symposium on Medical Robotics. 2017.
```



# Calibration parameters
```
Intrinsics: width, height, fx, fy, px, py. These apply to both left and right after rectification.
[ 384.	192.	373.47833252	373.47833252	182.91804504	113.72999573]
Extrinsics: rx, ry, rz, tx, ty, tz (in mm)
[ 0.	0.	0.	-5.63117313	0.	0.]
```
