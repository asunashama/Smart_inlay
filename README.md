# Smart_inlay
智能镶嵌式课程仓库

## 本小组选题为基于pytorch的简单验证码识别
主要识别数字＋字母类型的简单验证码
![](https://github.com/asunashama/Smart_inlay/tree/main/source/img/S747.jpg  "S747" )

## 关于卷积神经网络
* [神经网络](https://blog.csdn.net/illikang/article/details/82019945)
一个简单的网络示例：
![](https://github.com/asunashama/Smart_inlay/tree/main/source/img/em.jfif  "网络示例" )
全连接层(full-connected layer)，顾名思义，是将前面层的节点全部连接然后通过自己之后传入下一层。

卷积神经网络主要由这几类层构成：输入层、卷积层，ReLU层、池化（Pooling）层和全连接层（全连接层和常规神经网络中的一样）。通过将这些层叠加起来，就可以构建一个完整的卷积神经网络。在实际应用中往往将卷积层与ReLU层共同称之为卷积层。

### 卷积层
我们是使用卷积核来提取特征的，卷积核可以说是一个矩阵。卷积核的任务就如下所示：
![](https://github.com/asunashama/Smart_inlay/tree/main/source/img/juanjihe.gif  "卷积核" )
从左上角开始，卷积核就对应着数据的3*3的矩阵范围，然后相乘再相加得出一个值。按照这种顺序，每隔一个像素就操作一次，我们就可以得出9个值。这九个值形成的矩阵被我们称作激活映射（Activation map）。这就是我们的卷积层工作原理。

假使我们只有一个卷积核，那我们或许只能提取到一个边界。但假如我们有许多的卷积核检测不同的边界，不同的边界又构成不同的物体，这就是我们怎么从视觉图像检测物体的凭据了。所以，深度学习的“深”不仅仅是代表网络，也代表我们能检测的物体的深度。即越深，提取的特征也就越多。

### 池化层 (pooling layer)
池化层是降低参数，而降低参数的方法当然也只有删除参数了。池化层一般放在卷积层后面，所以池化层池化的是卷积层的输出。
![](https://github.com/asunashama/Smart_inlay/tree/main/source/img/chihua.png  "池化层" )
选择最大池化，应该是为了提取最明显的特征，所以选用的最大池化。平均池化就是顾及每一个像素，所以选择将所有的像素值都相加然后再平均。




## 关于pytorch
* [60分钟快速入门 PyTorch](https://zhuanlan.zhihu.com/p/66543791)
* [如何高效入门 PyTorch ？](https://zhuanlan.zhihu.com/p/96237032)
* [PyTorch 的基本使用](https://blog.csdn.net/YKenan/article/details/117163434)


### 什么是torch？
torch是火炬（确信
* [Pytorch与Torch的关系](https://zhuanlan.zhihu.com/p/438566725)



## 验证码生成 （
首先需要（在conda里面pip就行，注意不要把文件命名为`capchat.py`
``` python
pip install captcha #验证码模块
```
然后就能生成大量验证码图片了
###

## 参考资料
* [验证码生成](https://blog.csdn.net/qq_37781464/article/details/89919821)
* [卷积神经网络](https://blog.csdn.net/weixin_41417982/article/details/81412076)
* [卷积神经网络（CNN）详解](https://zhuanlan.zhihu.com/p/47184529)
* [对卷积神经网络工作原理直观的解释](https://www.zhihu.com/question/39022858/answer/224446917)