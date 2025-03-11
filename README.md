# 一个LBM项目的练习
项目使用taichi编写，快就完事了。源文件于openLBDEM。examples文件夹包含了示例文件。使用时，将main文件置于openLBDEM同级目录下，或者是修改导入目录。

1. main(Circular cylinder) 展示了圆柱绕流。 分别使用BGK, MRT, TRT 三种不同的碰撞模型，包含单位转换过程。

![圆柱绕流](results/main(Circular%20cylinder).gif)

2. main(sc1-Bubble or two-phase separation) 展示了使用原始shanchen模型模拟单组份两相分离的过程。该案例还可以进行拉普拉斯测试。

![单组份相分离](results/main(sc1-Bubble%20or%20two-phase%20separation).gif)

3. main(sc1-contact-line_) 展示了单组分中上下半平面的固体接触角。

![](results/main(sc1-contact-line_).gif)

3. main(sc2-Bubble or two-phase separation)展示了使用原始shanchen模型模拟多组分问题。

![多组分](results/main(sc2-Bubble%20or%20two-phase%20separation).gif)

也可以进行拉普拉斯测试

![](results/YLtest_2c.png#pic_center)
