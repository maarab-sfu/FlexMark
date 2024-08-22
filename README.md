# FlexMark: Adaptive Watermarking Method for Images
<details>
<summary>Publication detail:</summary>
  Authors: [M. A. Arab, A. Ghorbanpour, M. Hefeeda]

  Publication: [MMSys, 2024]

  DOI: [(https://doi.org/10.1145/3625468.364761)]

  Paper: [https://dl.acm.org/doi/pdf/10.1145/3625468.3647611]
</details>

## Overview
This repository contains the code and datasets associated with our paper titled "FlexMark: Adaptive Watermarking Method for Images". The paper introduces FlexMark, a robust and adaptive watermarking method for images, which achieves a better capacity-robustness trade-off than current methods and can easily be used for different applications. FlexMark categorizes and models the fundamental aspects of various image transformations, enabling it to achieve high accuracy in the presence of many practical transformations. FlexMark introduces new ideas to further improve the performance, including double-embedding of the input message, employing self-attention layers to identify the most suitable regions in the image to embed the watermark bits, and utilization of a discriminator to improve the visual quality of watermarked images. In addition, FlexMark offers a parameter, α, to enable users to control the trade-off between robustness and capacity to meet the requirements of different applications. We implement FlexMark and assess its performance using datasets commonly used in this domain. Our results show that FlexMark is robust against a wide range of image transformations, including ones that were never seen during its training, which shows its generality and practicality. Our results also show that FlexMark substantially outperforms the closest methods in the literature in terms of capacity and robustness.


