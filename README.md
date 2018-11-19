# pytorch-unflow
This is a personal reimplementation of UnFlow [1] using PyTorch. Should you be making use of this work, please cite the paper accordingly. Also, make sure to adhere to the <a href="https://github.com/simonmeister/UnFlow/blob/master/LICENSE">licensing terms</a> of the authors. Should you be making use of this particular implementation, please acknowledge it appropriately.

<a href="https://arxiv.org/abs/1711.07837" rel="Paper"><img src="http://www.arxiv-sanity.com/static/thumbs/1711.07837v1.pdf.jpg" alt="Paper" width="100%"></a>

For the original TensorFlow version of this work, please see: https://github.com/simonmeister/UnFlow
<br />
Another optical flow implementation from me: https://github.com/sniklaus/pytorch-pwc
<br />
And another optical flow implementation from me: https://github.com/sniklaus/pytorch-spynet

## setup
To download the pre-trained models, run `bash download.bash`. These originate from the original authors, I just converted them to PyTorch.

The correlation layer is implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided binary packages as outlined in the CuPy repository.

## usage
To run it on your own pair of images, use the following command. You can choose between two models, please make sure to see their paper / the code for more details.

```
python run.py --model css --first ./images/first.png --second ./images/second.png --out ./out.flo
```

I am afraid that I cannot guarantee that this reimplementation is correct. However, it produced results identical to the implementation of the original authors in the examples that I tried. Please feel free to contribute to this repository by submitting issues and pull requests.

## comparison
<p align="center"><img src="comparison/comparison.gif?raw=true" alt="Comparison"></p>

## license
As stated in the <a href="https://github.com/simonmeister/UnFlow/blob/master/LICENSE">licensing terms</a> of the authors of the paper, the models subject to an MIT license. Please make sure to further consult their licensing terms.

## references
```
[1]  @inproceedings{Meister_AAAI_2018,
         author = {Simon Meister and Junhwa Hur and Stefan Roth},
         title = {{UnFlow}: Unsupervised Learning of Optical Flow with a Bidirectional Census Loss},
         booktitle = {AAAI},
         year = {2018}
     }
```