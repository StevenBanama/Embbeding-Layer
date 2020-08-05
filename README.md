# A common layers for Embedding

In this repos, we relize commom version of embedding layer for face recognize, meanwhile we add parameters m4 to improve the performace.
In the cloth recognition, formula 1.2 improve the top 1 acc from 62.1% to 63.7% for resetnet50. 

<img src="https://raw.githubusercontent.com/StevenBanama/Embbeding-Layer/master/assets/formula_bedding.png">

The embedding space for iras dataset show as follow. Our plan got a more compat embedding, which not only reduce the distance of inner cluster, but also gain the inter distance in "arc-face expand". 
<div>
<img src="https://raw.githubusercontent.com/StevenBanama/Embbeding-Layer/master/assets/normface.png" width="200" height="200">
<img src="https://raw.githubusercontent.com/StevenBanama/Embbeding-Layer/master/assets/cosface.png" width="200" height="200">
<img src="https://raw.githubusercontent.com/StevenBanama/Embbeding-Layer/master/assets/arcface.png" width="200" height="200">
<img src="https://raw.githubusercontent.com/StevenBanama/Embbeding-Layer/master/assets/sphereface.png" width="200" height="200">
<img src="https://raw.githubusercontent.com/StevenBanama/Embbeding-Layer/master/assets/arcface_extend.png" width="200" height="200">
</div>

Reference:
  - https://arxiv.org/pdf/1801.07698 arcface(insight face)
  - https://arxiv.org/pdf/1801.09414 cosface
  - https://arxiv.org/pdf/1503.03832 am-face
  - https://arxiv.org/pdf/1704.08063 sphere-face
  - https://arxiv.org/pdf/2002.10857 circle-loss
