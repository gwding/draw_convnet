# draw_convnet

Python script for illustrating Convolutional Neural Network (ConvNet)

## Example image
![](https://raw.githubusercontent.com/gwding/draw_convnet/master/convnet_fig.png)

## Know issues
`Line2D` doesn't seem to work well under `python3 + matplotlib 2.0.0` as pointed out by @ahoereth 

The solution is to use follow these instructions.
```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements_python3.txt
```
