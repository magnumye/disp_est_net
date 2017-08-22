# disparity estimation network
CNN-based monocular disparity (inverse depth) estimation network for surgical videos collected in da Vinci surgery. The source code and data are associated with a [short report](https://arxiv.org/pdf/1705.08260.pdf) presented at the Hamlyn Symposium on Medical Robotics 2017. 

If you use the code or data, please cite following:

```
Ye, M., Johns, E., Handa, A., Zhang, L., Pratt, P. and Yang, G.Z., 2017. Self-Supervised Siamese Learning on Stereo Image Pairs for Depth Estimation in Robotic Surgery. Hamlyn Symposium on Medical Robotics. 2017.
 ```

You can download our [data]() and [pretrained models]() and place them in the "data" and "trained" folders, respectively.

### Prerequisites ###

Torch

[Torch-autograd](https://github.com/twitter/torch-autograd)

[gvnn](https://github.com/ankurhanda/gvnn)

[Torch-colormap](https://github.com/JannerM/torch-colormap) (for visualisation only)


### License ###

This code is distributed under BSD License.


### Notes ###

1. The autoencoder model in this implementation is slightly different from the one in the report. Certain layers have been removed for memory consideration and skip layers and multiscale training have been added.

2. Please adjust the mini-batch szie according to your specific GPU memory.

3. This implementation has been tested in Ubuntu.

4. Please see run_train.lua and run_inference.lua for example usage.

