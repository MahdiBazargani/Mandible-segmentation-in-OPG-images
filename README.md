# Mandible segmentation in OPG images

Mandible segmentation in OPG (Orthopantomography) images is the process of identifying and delineating the mandible bone from the surrounding tissues and structures in an OPG image.
In this project, neural network-based models are used for mandibular segmentation. The models reviewed include Unet, Unet++ and FPN.
# Dataset
In this study, an OPG dataset that includes panoramic dental X-rays of 116 patients has been used. These X-rays were obtained from the Noor Medical Imaging Center, Qom, Iran, and encompass a broad spectrum of dental conditions, ranging from healthy cases to partial and complete edentulism. Notably, the mandibles of all cases have been meticulously segmented by two certified dentists.
This dataset can be accessed through the link below.

https://data.mendeley.com/datasets/hxt48yk462/2

Out of 116 dataset samples, 80 cases were used for training, 20 cases were used for validation and 16 cases were used for testing.

The structure of the dataset is as follows.
```bash
├───test
│   ├───Images
│   ├───outputs
│   ├───Segmentation1
│   └───Segmentation2
└───train
    ├───Images
    ├───Segmentation1
    └───Segmentation2
```
# Preprocessing
The images are scaled down and the pixel values are normalized to the range 0 to 1.


# Train


```bash
$ python train.py -h

optional arguments:
  -h, --help         show this help message and exit
  --model            The architecture used, which can be unet, unet++ and FPN
  --epochs           Number of epochs
  --batch_size       Batch size
  --learning_rate    Learning rate
  --path_to_images   path to images
  --path_to_masks    path to masks
  --scale            Downscaling factor of the images

```
The following command can be used to train the network with the unet model.
```bash
python train.py
```
The diagram of loss and dice score for the unet model is as follows.

![unet loss](https://github.com/MahdiBazargani/Mandible-segmentation-in-OPG-images/blob/master/Figures/unet_loss.png?raw=true)

![unet dice score](https://github.com/MahdiBazargani/Mandible-segmentation-in-OPG-images/blob/master/Figures/unet_dice_score.png?raw=true)

The following command can be used to train the network with the Unet++ model.
```bash
python train.py --model unet++
```
The diagram of loss and dice score for the Unet++ model is as follows.

![unet++ loss](https://github.com/MahdiBazargani/Mandible-segmentation-in-OPG-images/blob/master/Figures/unetpp_loss.png?raw=true)

![unet++ dice score](https://github.com/MahdiBazargani/Mandible-segmentation-in-OPG-images/blob/master/Figures/unetpp_dice_score.png?raw=true)

The following command can be used to train the network with the FPN model.
```bash
python train.py --model FPN
```
The diagram of loss and dice score for the FPN model is as follows.

![SegFormer loss](https://github.com/MahdiBazargani/Mandible-segmentation-in-OPG-images/blob/master/Figures/FPN_loss.png?raw=true)

![SegFormer dice score](https://github.com/MahdiBazargani/Mandible-segmentation-in-OPG-images/blob/master/Figures/FPN_dice_score.png?raw=true)





# Test

```bash
$ python test.py -h

optional arguments:
  -h, --help         show this help message and exit
  --model            The architecture used, which can be unet, unet++ and FPN
  --checkpoint       path to model
  --path_to_images   path to images
  --path_to_masks    path to masks
  --output_dir       path to output directory
  --scale            Downscaling factor of the images

```

The value of dice score of Unet, Unet++ and FPN models for test data was 97.64%, 97.31% and 97.15% respectively.
According to the results, model Unet has the best performance among the 3 models.

The following figures show examples of the output of the models.


![unet outputs](https://github.com/MahdiBazargani/Mandible-segmentation-in-OPG-images/blob/master/Figures/unet_res.png?raw=true)

![unet++ outputs](https://github.com/MahdiBazargani/Mandible-segmentation-in-OPG-images/blob/master/Figures/unetpp_res.png?raw=true)

![FPN outputs](https://github.com/MahdiBazargani/Mandible-segmentation-in-OPG-images/blob/master/Figures/fpn_res.png?raw=true)


## Appendix

The structure of unet network is as follows.

![unet](https://github.com/MahdiBazargani/Mandible-segmentation-in-OPG-images/blob/master/Figures/unet.png?raw=true)

The structure of Unet++ network is as follows.

![Unet++](https://github.com/MahdiBazargani/Mandible-segmentation-in-OPG-images/blob/master/Figures/unetpp.jpg?raw=true)

The structure of FPN network is as follows.

![SegFormer](https://github.com/MahdiBazargani/Mandible-segmentation-in-OPG-images/blob/master/Figures/FPN.png?raw=true)



## License

[MIT](https://choosealicense.com/licenses/mit/)

