## Reparameterize YOLO-World

The reparameterization incorporates text embeddings as parameters into the model. For example, in the final classification layer, text embeddings are reparameterized into a simple 1x1 convolutional layer.

### Key Advantages from Reparameterization

> Reparameterized YOLO-World still has zero-shot ability!

* **Efficiency:** reparameterized YOLO-World has a simple and efficient archtecture, e.g., `conv1x1` is faster than `transpose & matmul`. In addition, it enables further optmization for deployment.
 
* **Accuracy:** reparameterized YOLO-World supports fine-tuning. Compared to the normal `fine-tuning` or `prompt tuning`, **reparameterized version can optimize the `neck` and `head` independently** since the `neck` and `head` have different parameters and do not depend on `text embeddings` anymore!
For example, fine-tuning the **reparameterized YOLO-World** obtains *46.3 AP* on COCO *val2017* while fine-tuning the normal version obtains *46.1 AP*, with all hyper-parameters kept the same.

### Getting Started

#### 1. Prepare cutstom text embeddings

You need to generate the text embeddings 


#### 2. Reparameterizing




#### 3. Prepare the model config

