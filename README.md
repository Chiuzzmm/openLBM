# LBM练习
示例全部使用taichi编写，LBM计算结构基本相同。部分小功能使用matlab计算。

1. BGK, MRT, TRT 展示了三种不同的碰撞模型以及相应的单位转换过程。
2. Maxwell 使用python和matlab计算了两种不同的状态方程。
3. shanchen-2PhaseMix 展示了使用原始shanchen模型模拟两相分离的过程。
4. Young-Laplace test 使用原始shanchen模型模拟无限域中的稳态液滴。
5. contact angle test  使用原始shanchen模型模拟了管道中的液滴。
6. contact angle test (曲面边界) 使用原始shanchen模型模拟了液滴撞击曲面固体的过程。
7. MRT-huang 使用Rongzong Huang 提出的模型模拟了无限域中的稳态液滴。
8. contact line on stationary circular cylinders 使用Rongzong Huang模型和BC边界模拟了上下半平面为两相的固体接触角。
9. IBM使用了浸没边界法模拟了静止颗粒在管道中的流场, IBM使用MDF格式。
10. IBM-SC使用浸没边界法代替固体边界，结合Rongzong Huang模型模拟两相流体。注意在该模拟中，IB节点设置在气体区域中需要适当的初始条件，否则容易崩溃。
