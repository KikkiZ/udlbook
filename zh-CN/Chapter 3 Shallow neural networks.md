# Chapter 3 Shallow neural networks

第2章介绍了使用一维线性回归模型进行有监督学习，然而，该模型只能将输入和输出关系描述为一条直线。本章将介绍**浅层神经网络**（shallow neural networks），它们可以描述分段的线性函数，并且足够灵活，可以近似多维输入和输出之间的任意复杂关系。

## 3.1 Neural network example

浅层神经网络是带有参数 $\phi$ 的函数 $y = f[x, \phi]$，它将多变量输入 $x$ 映射到多变量输出 $y$。我们将完整的定义推迟到[第3.4节](#3.4 Shallow neural network: general case)，在这里我们使用一个例子来介绍主要思想。网络 $f[x, \phi]$ 将标量输入 $x$ 映射到标量输出 $y$，并具有十个参数 $\phi = \{\phi_0, \phi_1, \phi_2, \phi_3, \theta_{10}, \theta_{11}, \theta_{20}, \theta_{21}, \theta_{30}, \theta_{31}\}$​：
$$
\begin{align}
y &= f[x, \theta]\\
&= \phi_0 + \phi_1a[\theta_{10} + \theta_{11}x] + \phi_2a[\theta_{20} + \theta_{21}x] + \phi_2a[\theta_{30} + \theta_{31}x]\tag{3.1}
\end{align}
$$
我们可以将这个算式分为三部分：首先，我们计算输入数据的三个线性函数（$\theta_{10} + \theta_{11}x$，$\theta_{20} + \theta_{21}x$，$\theta_{30} + \theta_{31}x$）。其次，我们将这三个结果传递给**激活函数**（activation function） $a[\bullet]$。最后，我们用 $\phi_1$、$\phi_2$ 和 $\phi_3$ 对第二步得到的激活值进行加权，将结果求和，并加上一个偏移量 $\phi_0$​。

为了完整的理解例子，我们必须定义激活函数 $a[\bullet]$。有很多种选择，但最常见的选择是**整流线性单元**（rectified linear unit，ReLU）：
$$
a[z] = ReLU[z] =
\begin{cases}
0\qquad z<0\\
z\qquad z\geq0
\end{cases}
\tag{3.2}
$$
当输入为正数时返回输入，否则返回0，函数图像如图3.1所示。

显然，我们不容易确定方程3.1表示哪一类输入-输出关系。尽管如此，上一章的思想仍然适用。方程3.1表示了一组函数，组中的特定成员取决于十个参数。如果我们知道这些参数，我们可以通过评估给定输入 $x$ 的方程来预测 $y$。给定训练数据集 $\{x_i, y_i\}^I_{i = 1}$，我们可以定义一个最小二乘损失函数 $L[\phi]$，并使用它来衡量模型在给定参数值 $\phi$ 下对数据集的描述效果。为了训练模型，我们需要寻找使该损失最小化的参数 $\hat{\phi}$​。

<img src="https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401291640616.jpg" alt="1" style="zoom:50%;" />

[图3.1 ReLU 这个激活函数输入如果小于0，则返回0，否则返回输入本身。换句话说，它将负值修剪为0。请注意，激活函数有许多其他可能的选择，详情见图3.13，但ReLU时最常用最容易理解的一个。]

![1](https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401291947743.jpg)

[图3.2 由方程3.1定义的函数组 a-c使在十个参数的三种不同选择下的函数，在每种情况下，输入-输出关系都是分段线性的。然而，每段连接的位置、连接之间的线性区域的斜率以及整体的高度是不同的。]

### 3.1.1 Neural network intuition

事实上，方程3.1代表了一系列连续分段线性函数，该函数最多具有四个线性区域，如图3.2所示。现在我们将分解方程3.1，并说明为什么它描述了这个函数。为了更容易理解，我们将把函数分为两个部分。首先，我们引入中间变量：
$$
h_1 = a[\theta_{10} + \theta_{11}x]\\
h_2 = a[\theta_{20} + \theta_{21}x]\\
h_3 = a[\theta_{30} + \theta_{31}x]
\tag{3.3}
$$
我们将 $h_1$、$h_2$ 和 $h_3$ 称为**隐藏单元**（hidden units）。然后，我们通过将这些隐藏单元与线性函数组合来计算输出：
$$
y = \theta_0 + \theta_1h_1 + \theta_2h_2 + \theta_3h_3\tag{3.4}
$$
图3.3展示了创建图3.2a中函数的计算流程。每个隐藏单元包含输入的线性函数 $\theta_{\bullet0} + \theta_{\bullet1}x$，并且该直线纵坐标小于0的部分被ReLU激活函数 $a[\bullet]$ 被剪裁。三条直线与 $x$ 轴的交点称为最终输出中的三个“关节”。然后，分别将这三条剪裁过的线加权为 $\phi_1$、$\phi_2$ 和 $\phi_3$。最后添加偏移量 $\phi_0$​，它控制着最终函数的整体高度。[Problems 3.2-3.8](#Problem)

![1](https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401292016813.jpg)

[图3.3 图3.2a中函数的计算过程 a-c）输入 $x$ 经过三个线性函数激活，每个函数具有不同的截距 $\theta_{\bullet0}$ 和斜率 $\theta_{\bullet}1$； d-f）每条线通过ReLU激活函数，将负值裁剪掉；g-i）三条被裁剪的线分别通过权重 $\phi_1$、$\phi_2$ 和 $\phi_3$ 进行加权计算； j）裁剪并加权的函数被求和，并添加一个控制高度的偏移量 $\phi_0$。四个线性区域对应于隐藏单元中不同的激活模式，在引用区域中，$h_2$ 是不活跃的，而 $h_1$ 与 $h_3$ 是活跃的。]

图3.3j中的每个线性区域对应与隐藏单元中不同的**激活模式**（activation pattern）。当一个单元被剪裁时，我们将其称为**不活跃的**（inactive），而当它未被裁剪时，我们将其称为**活跃的**（active）。例如，阴影区域接收来自于活跃的 $h_1$ 与 $h_3$，但不接受来自于不活跃的 $h_2$ 的贡献。每个线性区域的斜率由以下因素决定：i）该区域活跃的隐藏单元的原始斜率 $\theta_{\bullet1}$；ii）使用权重 $\phi_\bullet$ 与斜率相乘。例如，阴影区域的斜率为 $\theta_{11}\phi_1 + \theta_{31}\phi_3$​，其中第一项是图3.3g中的斜率，第二项是图3.3i中的斜率。

每个隐藏单元对函数贡献一个“关节”，因此对于三个隐藏单元，可以有四个线性区域。然而，这些区域中只有前三个斜率是独立的，第四个区域要么为0（所有隐藏单元都不活跃），要么是来自其他区域的斜率的和。[Problem 3.9](#Problems)

### 3.1.2 Depicting neural networks

我们一直在讨论具有一个输入、一个输出和三个隐藏单元的神经网络。我们在图3.4a中对这个网络进行了可视化，输入位于左侧，隐藏单元位于中间，输出位于右侧。每个连接代表十个参数之一。为了简化这个表示，通常我们不会标出参数，因此，这个网络通常被描述如图3.4b所示。

![1](https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401292043579.jpg)

[图3.4 描绘神经网络 a）输入 $x$ 位于左侧，隐藏单元 $h_1$、$h_2$ 和 $h_3$ 位于中间，输出 $y$ 位于右侧。计算从左到右进行，输入用于计算隐藏单元，它们组合起来创建输出。十个箭头每一个都代表一个参数，橙色为截距，黑色为斜率。每个参数都乘以源，并将结果添加到目标中。例如我们将参数 $\phi_1$ 乘以隐藏单元 $h_1$ 的输出并加到 $y$ 中。我们引入了一个额外的节点来，将偏移合并到输出中，因此我们将 $\phi_0$ 乘以1并添加到 $y$ 中。在隐藏单元中使用ReLU激活函数。 b）更常见的网络样式，省略了截距、ReLU函数和参数名。这种简化的样式描绘了相同的网络。]

## 3.2 Universal approximation theorem

在上一节中，我们介绍了一个具有一个输入、一个输出、使用ReLU激活函数和三个隐藏单元的神经网络。现在让我们在这个基础上略微推广一下，并考虑具有 $D$ 个隐藏单元的情况，其中第 $d$ 个隐藏单元用如下公式计算：
$$
h_d = a[\theta_{d0} + \theta_{d1}x]\tag{3.5}
$$
这些隐藏单元被线性组合以计算输出：
$$
y = \phi_0 + \sum^D_{d = 1}\phi_dh_d\tag{3.6}
$$
浅层神经网络中的隐藏单元的数量是对**网络容量**（network capacity）的一种度量。对于具有ReLU激活函数的网络，具有 $D$ 个隐藏单元的网络的输出最多具有 $D$ 个连接点，因此它是一个最多具有 $D + 1$​ 个线性区域的分段函数。随着我们添加更多的隐藏单元，模型可以逼近更复杂的函数。[Problem 3.10](#Problems)

事实上，只要有足够的容量，浅层神经网络就可以以任意精度描述定义在实线子集上的任何连续一维函数。为了理解这一点，请考虑每当为网络添加一个隐藏单元时，我们都为函数添加了另一个线性区域。随着线性区域变得更多，它们代表了函数越来越小的部分，这些部分可以越来越好的近似为一条直线，如图3.5所示。神经网络逼近任何连续函数的能力可以被正式的证明，这也被称为**通用逼近定理**（universal approximation theorem）。

![1](https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401292200688.jpg)

[图3.5 通过分段线性模型对一维函数的逼近 图a-c随着区域数量增加，模型越来越接近连续函数。具有标量输入的神经网络每个创建一个额外的线性区域。通用逼近定理证明，通过足够多的隐藏单元，浅层神经网络可以以任意精度描述定义在 $\R^D$ 的子集上的任何连续函数。]

## 3.3 Mutivariate inputs and outputs

在上面的例子里，网络具有单个标量输入 $x$ 和单个标量输出 $y$。然而，通用逼近定理也适用于更一般的情况，即网络将多元输入 $x = [x_1, x_2, ..., x_{D_i}]^T$ 映射到多元输出预测 $y = [y_1, y_2, ...,y_{D_i}]^T$。本节将探讨如何拓展模型以预测多元输出，然后再考虑多元输入。最后，在3.4节中，我们将介绍一个浅层神经网络的通用定义。

### 3.3.1 Visualizing multivariate outputs

为了将网络拓展到多元输出 $y$，我们仅需为每个输出使用不同的隐藏单元的线性函数。因此，具有一个标量输入 $x$、4个隐藏单元 $h_1$、$h_2$、$h_3$ 和 $h_4$ 以及2D多元输出 $y = [y_1, y_2]^T$​ 的网络的隐藏单元将定义为：
$$
h_1 = a[\theta_{10} + \theta{11}x]\\
h_2 = a[\theta_{20} + \theta{21}x]\\
h_3 = a[\theta_{30} + \theta{31}x]\\
h_4 = a[\theta_{40} + \theta{41}x]
\tag{3.7}
$$
网络输出结果为：
$$
y_1 = \phi_{10} + \phi_{11}h_1 + \phi_{12}h_2 + \phi_{13}h_3 + \phi_{14}h_4\\
y_2 = \phi_{20} + \phi_{21}h_1 + \phi_{22}h_2 + \phi_{23}h_3 + \phi_{24}h_4
\tag{3.8}
$$
这两个输出是4个隐藏单元的两个不同的线性组合。

如图3.3所示，分段函数中的转折点取决于初始线性函数 $\theta_{\bullet0} + \theta_{\bullet1}x$ 在隐藏单元中被ReLU函数 $a[\bullet]$ 截断的位置。由于输出 $y_1$ 和 $y_2$ 都是相同的四个隐藏单元的不同线性函数，每个函数中的四个拐点必须位于相同的位置。然而，线性区域的斜率和整体的垂直偏移可能不同，如图3.6所示。[Problem 3.11](#Problems)

![1](https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401301524905.jpg)

[图3.6 具有一个输入、四个隐藏单元和两个输出的网络 a）网络结构的可视化。 b）该网络产生两个分段线性函数 $y_1[x]$ 和 $y_2[x]$。这些函数的四个转折点被限制在同一个位置，因为它们共享相同的隐藏单元，但斜率和整体高度可能不同。]

### 3.3.2 Visualizing multivariate inputs

为了处理多元输入 $x$，我们扩展了输入和隐藏单元之间的线性关系。因此，具有两个输入 $x = [x_1, x_2]^T$ 、三个隐藏单元和一个标量输出的网络结构如图3.7所示。三个隐藏单元定义为：
$$
h_1 = a[\theta_{10} + \theta_{11}x_1 + \theta_{12}x_2]\\
h_2 = a[\theta_{20} + \theta_{21}x_1 + \theta_{22}x_2]\\
h_3 = a[\theta_{30} + \theta_{31}x_1 + \theta_{32}x_2]
\tag{3.9}
$$
现在，每个输入都有一个斜率参数，隐藏单元则以常规的方式组合形成输出：
$$
y = \phi_0 + \phi_1h_1 + \phi_2h_2 + \phi_3h_3\tag{3.10}
$$
<img src="https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401301604798.jpeg" alt="1" style="zoom:50%;" />

[图3.7 具有2维多元输入 $x = [x_1, x_2]^T$ 和标量输出 $y$ 的神经网络可视化]

图3.8说明了该网络的处理过程，每个隐藏单元接收两个输入的线性组合，形成了3D输入-输出空间中的定向平面。激活函数将这些平面的负值剪切为0。然后在等式3.10中重新组合剪切后的平面，以创建由凸多边形区域形成的连续分段线性曲面，如图3.8j所示。每个区域对应不同的激活模式，例如在中间三角区域中，第一个和第三个隐藏单元处于激活状态，而第二个隐藏单元处于非激活状态。[Problems 3.12-3.13](#Problems)

![1](https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401301606470.jpg)

[图3.8 对具有两个输入 $x = [x_1, x_2]^T$，三个隐藏单元 $h_1$、$h_2$、$h_3$和一个输出 $y$ 的网络进行处理 a-c）每个隐藏单元的输入是两个输入的线性函数，依次对应于一个平面。亮度表示函数的输出。例如，在图a中，亮度表示 $\theta_{10} + \theta_{11}x_1 + \theta_{12}x_2$​，细线是等高线。 d-f）每个平面都通过ReLU激活函数进行剪裁，青色线是转折点。 g-i）对剪切后的平面进行加权。 j）与确定曲面整体高度的偏移量相加，结果是由凸多边形区域组成连续曲面。]

当模型有两个以上的输入时，它将变得难以可视化。但是，解释方式是类似的，输出将是输入的连续分段线性函数，其中线性区域现在是多维输入空间中的凸多面体。

值得注意的是，随着输入维度的增加，线性区域的数量迅速增加，如图3.9所示。为了感受增长速度，我们为每个隐藏单元定义一个超平面，将空间分为该单元所在的部分和不在的部分，如图3.8d-f中所示。如果我没有与输入维度 $D_i$ 相同数量的隐藏的那元，我们可以将每个超平面与坐标轴之一对齐，如图3.10所示。对于两个输入维度，这将把空间划分成四个象限；对于三个维度，这将空间划分为八个象限。对于 $D_i$ 个维度，网络将会创建 $2^{D_i}$ 个区域。浅层神经网络通常具有比输入维度更多的隐藏单元，因此它们通常会创建远超于 $2^{D_i}$​ 个线性区域。

![1](https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401301713672.jpeg)

[图3.9 线性区域与隐藏单元 a）对五个不同的输入维度 $D_i = \{1, 5, 10, 50, 100\}$，最大可能区域作为隐藏单元数量的函数进行绘制。在高维空间中，区域数量迅速增加；当 $D = 500$ 个单元且输入大小 $D_i = 100$ 时，可能超过存在10107个区域。 b）相同的数据作为参数数量的函数进行绘制。实心点代表与图a中相同的模型，具有 $D = 500$​ 个隐藏单元。该网络具有51,001个参数，而这现在被认为是非常小的。]

![1](https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401301726996.jpg)

[图3.10 线性区域数量与输入维度 a）对于单个输入维度，具有一个隐藏单元的模型会产生一个转折点，将轴分成两个线性区域。 b）对于两个输入维度，具有两个隐藏单元的模型可以使用两条线将输入空间划分为四个象限。 c）对于三个输入维度，具有三个隐藏单元的模型可以使用三个平面将输入空间划分为8个区域。按照这个论点，可以得出具有 $D_i$ 个输入维度和 $D_i$ 个隐藏单元的模型可以使用 $D_i$ 个超平面将输入空间划分为 $2^{D_i}$ 个线性区域的结论。]

## 3.4 Shallow neural networks: general case

我们描述了几个浅层神经网络的示例，以帮助我们更好的理解它们是如何工作的。现在，我们将为浅层神经网络定义一个通用方程 $y = f[x, \phi]$，该方程使用 $h\in\R^D$ 个隐藏单元将多维输入 $x\in\R^{D_i}$ 映射到多维输出 $y\in\R^{D_o}$。每个隐藏单元计算如下：
$$
h_d = a[\theta_{d0} + \sum^{D_i}_{i = 1}\theta_{di}x_i]\tag{3.11}
$$
这些隐藏单元被线性组合计算出输出：
$$
y_j = \phi_{j0} + \sum^D_{d = 1}\phi_{jd}h_d\tag{3.12}
$$
其中 $a[\bullet]$ 是非线性激活函数，该模型具有参数 $\phi = \{\theta_{\bullet\bullet}, \phi_{\bullet\bullet}\}$​。图3.11显示了一个具有三个输入、三个隐藏单元和两个输出的示例。[Problems 3.14-3.17](#Problems)

<img src="https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401301903008.jpg" alt="1" style="zoom:50%;" />

[图3.11具有三个输入和两个输出的神经网络可视化 该网络有20个参数，有15个斜率和5个偏移量。]

激活函数允许模型描述输入和输出之间的非线性关系，因此它本身必须是非线性的。如果没有激活函数或使用线性激活函数，整体的输入-输出的映射将被限制为线性。研究者已经尝试了许多不同的激活函数，如图3.13所示，但最常见的选择是之前使用的ReLU，它具有易于解释的优点。使用ReLU激活函数，网络将输入空间划分为ReLU函数中的转折点计算得到的超平面的交集定义的凸多面体，每个凸多面体包含一个不同的线性函数。对于每个输出来说，凸多面体是相同的，但他们包含的线性函数可能不同。

## 3.5 Terminology

神经网络有很多相关的术语，我们将通过引入一些术语来结束本章。它们通常用**层**（layer）来称呼，图3.12的左侧是**输入层**（input layer），中间是**隐藏层**（hidden layer），右侧是**输出层**（output layer），因此我们可以说图3.12中的网络有一个包含四个隐藏单元的隐藏层。隐藏单元有时候被称为**神经元**（neurons）。当我们通过神经网络传递数据时，隐藏层输出的值在被激活函数激活前称为**预激活**（pre-activations），通过隐藏层激活函数的值称为**激活**（activations）。

由于历史原因，至少有一层隐藏层的神经网络也被称为**多层感知器**（multi-layer perception，MLP）。具有一个隐藏层的网络被称为**浅层神经网络**（shallow neural networks）；具有多个隐藏层的网络被称为**深度神经网络**（deep neural networks）。神经元连接形成非循环图的神经网络称为**前馈网络**（feed-forward networks）。如果一层中的每个神经元都连接到下一层中的每个神经元时则称该网络是**全连接**（fully connected）的。这些连接代表隐藏的方程中的斜率参数被称为**网络权重**（network weight）。偏移参数被称为**偏差**（bias）。

![1](https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401301944241.jpg)

[图3.12 术语 浅层网络由输入层、隐藏层和输出层组成，每一层通过前向连接与下一层相连，因此，这些模型被称为前馈网络。当一层中的每个变量都连接到下一层的变量时，我们称为全连接网络。每个连接代表隐藏的方程中的斜率参数，这些参数被称为权重。隐藏层的变量称为神经元或隐藏单元。隐藏层输出的值在被激活函数激活前称为预激活，通过隐藏层激活函数的值称为激活。]

## 3.6 Summary

浅层神经网络具有一层隐藏层，它们通过以下步骤进行计算：i）对输入计算多个线性函数；ii）将每个结果通过激活函数进行处理；iii）通过这些激活函数的线性组合来形成输出。浅层神经网络基于输入 $x$ 对预测 $y$ 进行分割，将输入空间划分为一种分段线性区域的连续曲面。通过足够数量的神经元，浅层神经网络可以以任意精度逼近任何连续函数。

第4章将讨论深度神经网络，它通过添加更多的隐藏层来扩展本章的模型。第5-7章描述了如何训练这些模型。

## Notes

**神经网络**：如果本章使用的模型只是函数，那么为什么称它们为神经网络？这种联系很薄弱的。如图3.12所示的可视化由彼此紧密连接的节点组成。这与哺乳动物大脑的神经元看起来有相似之处，因为它们也有密集的连接。然而，几乎没有证据表明大脑的计算方式与神经网络相同，并且思考生物学的未来对深度学习并没有帮助。

**神经网络的发展史**：McCulloch和Pitts于1943年首次提出人工神经元的概念[^1]，它将输入结合起来产生输出，但这个模型没有实际的学习算法。Rosenblatt于1958年开发了**感知器**（perceptron），它线性地组合输入，然后对它们进行阈值处理以做出是或否的决策[^2]。他还提供了一种从数据中学习权重的算法。Minsky和Paoert于1969年认为线性函数对于一般的分类问题是不够的，但是添加具有非线性激活函数的隐藏层可以允许学习更宽的输入-输出关系[^3]。然而，他们得出结论认为Rosenblatt的算法不能学习这类模型的参数，直到20世纪80年代才开发出了实用的算法（详情请看第7章），同时神经网络的重要工作重新启动。神经网络的发展史由Kurenkov[^4]、Sejnowski[^5]和Schmidhuber[^6]进行了记录。

**激活函数**：ReLU函数最早在Fukushima时就已经被使用了[^7]。然而，在神经网络发展的早期阶段，更常使用的是Sigmoid或tanh函数，如图3.13a所示。ReLU由Jarrett[^8]、Nari和Hinton[^9]以及Glorot[^10]的研究而重新受到关注，并成为现代神经网络成功的重要组成部分。它具有输出对输入的导数在输入大于零时始终为1的特性，这有助于提升训练的稳定性和效率，更多请看第7章。并在于Sigmoid激活函数的导数形成对比，后者在绝对值较大的输入时非常接近于0。

然而，ReLU函数的缺点是，对于负输入的导数为0。如果所有的训练样本对给定的ReLU函数生成负输入，那么我们在训练过程中无法改进输入到该ReLU的参数。对于传入权重的梯度在局部时平坦的，因此我们无法“下坡”，这被称为ReLU的死亡问题。为了解决这个问题，提出了许多ReLU的变体，如图3.13b所示，包括：i）Maas提出leaky ReLU[^11]，他对于负值也有线性输出，但斜率较小，为0.1；ii）He提出parametric ReLU[^12]，它将负半轴的斜率视为未知参数；iii）Shang提出concatenate ReLU[^13]，它产生两个输出，其中一个类似于传统的ReLU在0以下截断，另一个在0以上截断。

还有各种平滑函数的研究，如图3.13c-d所示，其中包括Glorot提出的softplus函数[^14]、Hendrycks和Gimpel提出的高斯线性误差单元[^15]、sigmoid线性单元[^16]以及Clevert提出的指数线性单元[^17]。这些函数大部分都是为了避免神经元死亡问题，同时限制负值的梯度。Klambauer引入了缩放指数线性单元[^18]，如图3.13e所示，这个函数在输入方差有限范围时有助于稳定激活的方差。Ramachandran采用经验方法选择激活函数[^19]。他们在可能的函数空间中搜索，找到在各种监督学习任务上表现最好的函数。最优函数被发现是 $a[x] = \frac{x}{1 + e^{-\beta x}}$​，其中β是一个学习参数，如图3.13f所示。他们将这个函数称为Swish。有趣的是，这其实是基于之前Hendrycks和Gimpel[^20]以及Elfwing[^21]提出的激活函数的重新发现。Howard用HardSwish函数来近似Swish函数[^22]，它的形状非常相似但计算速度更快：
$$
HardSwish[z]=
\begin{cases}
0 &z < -3\\
\frac{z(z + 3)}{6} &-3\leq z\leq3\\
z &z > 3
\end{cases}
\tag{3.13}
$$
关于这些激活函数哪个在实践中更优的问题，没有明确的答案。然而，leaky ReLU、parameterized ReLU和许多连续函数在特定情况下可以显示出比ReLU稍微好一些的性能提升。在本书的剩余部分，我们将注意力集中在具有基本ReLU函数的神经网络上，因为可以通过线性区域的数量来描述它们所创建的函数。

![1](https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401302124064.jpg)

[图3.13 激活函数 a) sigmoid和tanh函数。 b) leaky ReLU和parametric ReLU，参数为0.25。 c) SoftPlus、高斯误差线性单元和sigmoid线性单元。d) 参数为0.5和1.0的指数线性单元。e) 缩放指数线性单元。f) 参数为0.4、1.0和1.4的Swish激活函数。]

**通用逼近定理**：该定理的width version指出，一个具有包含有限个隐藏单元和激活函数的隐藏层的网络可以以任意精度逼近定义在 $\R^n$​ 的子集上的任何连续函数。这是由Cybenko针对Sigmoid激活函数证明的[^23]，并后来被证明对更大类别的非线性激活函数也成立[^24]。

**线性区域的数量**：考虑一个具有 $D_i$ 大于2维输入和 $D$ 个隐藏单元的浅层神经网络。线性区域的数量由ReLU函数中“转折点”所创建的 $D$ 个超平面的交点确定，如图3.8d-f所示。每个区域是由不同组合的ReLU函数是否对输入进行修剪而创建的。在 $D_i \leq D$ 维输入空间中，由 $D$ 个超平面创建的区域数量被Zaslavsky证明最多为 $\sum^{D_i}_{j = 0}(D_j)$[^25]。一般而言，浅层神经网络的隐藏单元数量 $D$ 几乎总是大于输入维度 $D_i$，并创建了 $2^{D_i}$ 到 $2^D$​ 个线性区域之间的数量。

**线性、仿射（affine）和非线性函数**：从技术上讲，线性变换 $f[\bullet]$ 是指任何遵守叠加原理的函数，即 $f[a + b] = f[a] + f[b]$，这一定义表明了 $f[2a] = 2f[a]$。加权和 $f[h_1, h_2, h_3] = \phi_1h_1 + \phi_2h_2 + \phi_3h_3$ 是线性的，但一旦增加了偏差，即$f[h_1, h_2, h_3] = \phi_0 + \phi_1h_1 + \phi_2h_2 + \phi_3h_3$，这一定义将不再成立。考虑到当我们将前一个函数的参数加倍时，输出也会加倍。而对于后一个函数，情况并非如此，它更恰当地被称为仿射函数。然而，在机器学习中，通常会混淆这些术语。在本书中，我们遵循这种惯例，并将两者都称为线性函数。我们将遇到的所有其他函数都是非线性的。

## Problems

* **Problem 3.1**：如果方程3.1中的激活函数是线性的，即 $a[z] = \phi_0 + \phi_1z$，那么从输入到输出的映射将是线性的吗？如果移除激活函数，即 $a[z] = z$，那么将得到一个简单的线性映射，输入和输出之间的关系将是直接的一对一映射吗？
* **Problem 3.2**：对于图3.3j中的四个线性区域，请指出哪些隐藏单元是非活动的，哪些是活动的。
* **Problem 3.3**：在图3.3j中，根据十个参数 $\phi$ 和输入 $x$，推导出函数“关节”位置的表达式，推导出四个线性区域的斜率表达式。
* **Problem 3.4**：请画出类似于图3.3的图像，其中第三个隐藏单元的y截距和斜率已经发生了变化，就像图3.14c中那样。假设其它参数保持不变。

![1](https://raw.githubusercontent.com/KikkiZ/ChartBed/main/typora/202401302154775.jpg)

[图3.14 Problem3.4中，一个输入、三个隐藏单元和一个输出的网络处理 a-c）每个隐藏单元的输入是输入的线性函数。前两个与图3.3中的相同，但最后一个不同。]

* **Problem 3.5**：证明对于 $\alpha \in \R^+$，以下性质成立：

$$
ReLU[\alpha z] = \alpha ReLU[z]\tag{3.14}
$$

这被称为ReLU函数的**非负齐次**性质（non-negative homegeneity）。

* **Problem 3.6**：在Problem 3.5的基础上，当我们将参数 $\theta_{10}$ 和 $\theta_{11}$ 乘以正常数 $\alpha$，并将斜率 $\phi_1$ 除以相同的参数 $\alpha$ 时，方程3.3和3.4中定义的浅层网络会发生什么？如果 $\alpha$ 是负数会发生什么？
* **Problem 3.7**：考虑使用最小二乘损失函数拟合方程3.1中的模型，这个损失函数是否有唯一的最小值？或者说，是否存在单一的“最佳”参数集？
* **Problem 3.8**：考虑将ReLU激活函数替换为：i）海维赛德阶跃函数 $heaviside[z]$；ii）双曲正切函数$tanh[z]$；iii）矩形函数 $rect[z]$；iv）正弦函数 $sin[z]$​，其中：

$$
heaviside[z] = 
\begin{cases}
0\qquad z < 0\\
1\qquad z \geq 0
\end{cases}
\tag{3.15}
$$
$$
rect[z] = 
\begin{cases}
0\qquad z < 0\\
1\qquad 0 \leq z \leq 1\\
0\qquad z > 1
\end{cases}
\tag{3.16}
$$

重新为每个激活函数绘制图3.3图像。原始参数为 $\phi = \{\phi_0, \phi_1, \phi_2, \phi_3, \theta_{10}, \theta_{11}, \theta_{20}, \theta_{21}, \theta_{30}, \theta_{31}\} = \{−0.23, −1.3, 1.3, 0.66, −0.2, 0.4, −0.9, 0.9, 1.1, −0.7\}$。提供一个非正式描述：神经网络以每种激活函数为特征，包括一个输入、三个隐藏单元和一个输出，可以创建的函数族。

* **Problem 3.9**：证明图3.3中第三个线性区域的斜率是第一和第四个线性区域斜率之和。
* **Problem 3.10**：考虑一个具有一个输入、一个输出和三个隐藏单元的神经网络，如图3.3中的结构展示了这样一个神经网络可以创建四个线性区域的情况。在什么情况下，这个网络可以产生少于四个线性区域的函数呢？
* **Problem 3.11**：图3.6中的模型共有多少个参数？
* **Problem 3.12**：图3.7中的模型共有多少个参数？
* **Problem 3.13**：图3.8中的七个区域的激活模式是什么？换句话说，对于每个区域，哪些隐藏单元处于激活状态，哪些处于非激活状态？
* **Problem 3.14**：请提供图3.11中定义网络的方程式，应该有三个方程式用于计算来自输入的三个隐藏单元，以及两个方程式用于计算来自隐藏单元的输出。
* **Problem 3.15**：图3.11中的网络能够创建的最大可能的3D线性区域数量是多少？
* **Problem 3.16**：为拥有两个输入、四个隐藏单元和三个输出的神经网络写出方程式，并按照图3.11的风格绘制该模型。
* **Problem 3.17**：方程3.11和3.12定义了一个具有 $D_i$ 个输入、包含 $D$ 个隐藏单元的隐藏层和 $D_o$ 个输出的通用神经网络。以 $D_i$、$D$ 和 $D_o$ 为参数，找出模型中参数数量的表达式。
* **Problem 3.18**：证明具有 $D_i$ = 2维输入，$D_o$ = 1维输出和 $D$ = 3个隐藏单元的浅层网络所创建的最大区域数量为七，如图3.8j所示。使用Zaslavsky的结果，即通过用 $D$ 个超平面划分 $D_i$ 维空间所创建的最大区域数量为 $\sum^{D_i}_{j = 0}D_j$。如果我们向这个模型添加两个隐藏单元，使 $D$ = 5，那么最大区域数量是多少？

## Reference

[^1]: McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. The Bulletin of Mathematical Biophysics, 5(4), 115–133.
[^2]: Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. Psychological review, 65(6), 386.
[^3]: Minsky, M., & Papert, S. A. (1969). Perceptrons: An introduction to computational geometry. MIT Press.
[^4]: Kurenkov, A. (2020). A Brief History of Neural Nets and Deep Learning. https://www.skynettoday.com/overviews/neural-net-history.
[^5]: Sejnowski, T. J. (2018). The deep learning revolution. MIT press.
[^6]: Schmidhuber, J. (2022). Annotated history of modern AI and deep learning. arXiv:2212.11279.
[^7]: Fukushima, K. (1969). Visual feature extraction by a multilayered network of analog threshold elements. IEEE Transactions on Systems Science and Cybernetics, 5(4), 322–333.
[^8]: Jarrett, K., Kavukcuoglu, K., Ranzato, M., & LeCun, Y. (2009). What is the best multi-stage architecture for object recognition? IEEE International Conference on Computer Vision, 2146–2153.
[^9]: Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. International Conference on Machine Learning, 807–814.
[^10]: Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. International Conference on Artificial Intelligence and Statistics, 315–323.
[^11]: Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013). Rectifier nonlinearities improve neural network acoustic models. ICML Workshop on Deep Learning for Audio, Speech, and Language Processing.
[^12]: He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing humanlevel performance on ImageNet classification. IEEE International Conference on Computer Vision, 1026–1034.
[^13]: Shang, W., Sohn, K., Almeida, D., & Lee, H. (2016). Understanding and improving convolutional neural networks via concatenated rectified linear units. International Conference on Machine Learning, 2217–2225.
[^14]: Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. International Conference on Artificial Intelligence and Statistics, 315–323.
[^15]: Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). arXiv:1606.08415.
[^16]: Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). arXiv:1606.08415.
[^17]: Clevert, D.-A., Unterthiner, T., & Hochreiter, S. (2015). Fast and accurate deep network learning by exponential linear units (ELUs). arXiv:1511.07289.
[^18]: Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017). Self-normalizing neural networks. Neural Information Processing Systems, 972–981.
[^19]: Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for activation functions. arXiv:1710.05941.
[^20]: Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). arXiv:1606.08415.
[^21]: Elfwing, S., Uchibe, E., & Doya, K. (2018). Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107 , 3–11.
[^22]: Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., et al. (2019). Searching for MobileNetV3. IEEE/CVF International Conference on Computer Vision, 1314–1324.
[^23]: Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems, 2(4), 303–314.
[^24]: Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. Neural Networks, 4(2), 251–257.
[^25]: Zaslavsky, T. (1975). Facing up to arrangements: Face-count formulas for partitions of space by hyperplanes: Face-count formulas for partitions of space by hyperplanes. Memoirs of the American Mathematical Society.
