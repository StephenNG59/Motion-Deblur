# Notes
* 目前初始的核需要是中心对称的……
* 楼上傻逼，搞错了correlation和convolution了。频域相乘是真.convolution，但是图像处理是correlation。只有中心对称的时候两者才相同。
* 另外，频域相乘的结果相当于在四周pad了kernel size - 1的一圈0，当kernel size >= 3时，频域相乘结果会比卷积结果更宽更高。
* opencv的imread读取通道是bgr，matlabplot是rgb。


# Todo
-[ ] psi的update似乎和原先没有差别？
-[ ] E(f)的研究