# FULLY-CONNECTED-LAYER
学习总结
1.首先是transform的作用：一般用于图像预处理，包括随机裁剪之类的
2.mistgpu服务器使用指南：继承全局站点包，才能利用上预装软件框架
3.batchsize和output的数量差了x倍，这里一般是模型出错了，例如卷积层和全连接层之间的问题
4.每一层卷积都要经过 加工-激活-池化
5.self.AA2D = nn.AdaptiveAvgPool2d(1)这个全局自动平均池化 非常万能
例如说进去的是8x3x224x224，用这个函数可以变成8x3x1x1（因为是2d池化）
6.通过torch.squeeze（x）函数，可以把张量为1的维度压缩掉，但是不能用squeeze（1），这样只能压缩一个维度，不能把两个为1的维度都压缩掉
7.随机裁剪+resize是有效的预处理方式
8.transforms.ToTensor(),  # rgb归一化
9.优化器sche还在学习中
10.mseloss似乎比crossentropy轻视张量一些，下次试一试
11.不许乱用x.view函数！！！！！！！！！！！！
