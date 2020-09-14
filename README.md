# Steganography_GANs
Image steganography usings Generative Adverserial Networks


**Abstract**

Steganography is a generic term that denotes all those techniques that somehow try to hide information within other forms of data.  Differently from cryptography, which aims to hide messages by manipulation of the data, steganography aims to hide the existence of the information itself.

In this paper we are going to present steganography applied to images and audio, and in particular we will analyze the benefits that generative adversarial training produces in this context. The method used consists in three networks which works toghether: the first, which from now on we will refer to as \emph{encoder}, responsible for hiding the information, the second, named \emph{decoder}, responsible for recovering the secret message and the third, called the \emph{critic} which detect the presence of the hided information. The real place in which the adversarial training takes place is between the encoder and the critic, this last one provides feedback on the performance of the second and ensure the encoder to produce realistic images as much as possible. 

In this project we re-implemented and extended the work done by Kevin A. Zhang, Alfredo Cuesta-Infante, Lei Xu and Kalyan Veeramachaneni, in this paper:[Stegano-GAN: High Capacity Image Steganography with GANs](https://arxiv.org/abs/1901.03892), and by $Dengpan Ye$, $Shunzhi Jiang$, and $Jiaqin Huang$ in this paper [Heard More Than Heard: An Audio Steganography Method Based on GAN](https://arxiv.org/abs/1907.04986).

*Keywords: Image Steganography, Audio Steganography, Information Hiding, Unsupervised Learning, Deep Learning, Generative Adversarial Network.*

<p align="center" width="100%">
<img src="/images/img1.png" alt="" width= '100%'/>
</p>

<mark>The project has been developed under the supervision of *Prof. Danilo Comminiello*. The work is required for the fulfilment of credits for the *Neural Networks course* given by *Prof. Aurelio Uncini*, A.Y. 2019-2020. </mark>

## Documentation

If you want to read more about the project developing, you can find all the details in the [report](https://github.com/garg-akash/Steganography_GANs/blob/master/report.pdf).

## Datasets used

**Image:**

<p align="center" width="100%">
<img src="/images/img2.png" alt="" width= '100%'/>
</p>

You can download the test dataset [here](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip). This validation dataset is taken from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) by *Eirikur Agustsson and Radu Timofte. "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study".  Proceedings of the CVPR 2017 Workshop (2017)*


**Audio:** 

<p align="center" width="100%">
<img src="/images/aud.png" alt="" width= '100%'/>
</p>

You can download the test dataset [here](https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_test.zip?download=1). This validation dataset is taken from [FSDKaggle2018](https://zenodo.org/record/2552860#.XyExny2ub_T) by *Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel P. W. Ellis, Xavier Favory, Jordi Pons, Xavier Serra. "General-purpose Tagging of Freesound Audio with AudioSet Labels: Task Description, Dataset, and Baseline". Proceedings of the DCASE 2018 Workshop (2018)*
