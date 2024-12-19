---
layout: default
title: Quantization Aware Training
description: In-depth guide to implementing quantization aware training in PyTorch
date: 2024-03-21
---

# Quantization Aware Training

In this tutorial, we will be looking at the concept of quantization aware training, how does it work in depth, its benefits and how to implement it in PyTorch. To properly understand
this concept, proper understanding of quantization basics is required.


## What is Quantization?

Formally, quantization is the process of constraining an input from a continuous or otherwise large set of values to a discrete set of values. You can think of it as a way to reduce the precision of the data. In neural networks, quantization is the process of reducing the precision of the weights and activations. This can be helpful in different ways.

1) **Memory Reduction:**
    In the example of current LLMs, the weights of the feed-forward layers are quite large. Imagine a forward layer weight matrix in Llama 3 70B Model, the weight matrix could be of size
    `8192 * 8192`. In case of `float16`, this weight matrix would require `8192 * 8192 * 2 = 134,217,728` bytes of memory (approximately 128 MB). This is a lot of memory to store and process, when we consider the fact that the model has multiple such layers. In case we reduced
    the precision of the weights, we can reduce the load times from memory approximately two, four-fold respectively when using `int8` or `int4` data types.

2) **Speedup:**
    Quantization can also help in speeding up the inference, sometimes even the training process. From computer architecture perspective, the operations on large data types, such as
    `float16` or `float32`are expensive and slow. These operations are way faster and cheaper when performed on smaller data types like `int8` or `float8`. When we take a look at the current state-of-the-art GPU Nvidia H100 and its [datasheet](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet), we can see that the performance of the GPU Tensor Cores linearly increases with the decrease in the data type size.


Now, that we have a basic understanding on why quantization is important, let's take a look at how quantization works. 

***Important*** To simplify things, we will only be looking at quantization to a lower precision data type, that exists in the PyTorch framework, to avoid hassles of binary operations. That is, from `torch.float16` to `torch.int8`. Also, we will be considering a method called `Linear Quantization`. This is the most common method of quantization.

### How does Quantization work?
If we think of `int8` as a data type, it can store values in the range of `[-128, 127]`. However, our weights and activations in `float16` have a range of `[-65504, 65504]`. Also, this range in `float16` is not uniformly distributed, therefore accommodating a lot more possible values. To quantize the weights and activations, we need to map the values in `float16` to the range of `int8`. This can be done by the following steps:

1) **Min and Max Calculation:**
    We need to calculate the minimum and maximum values in the data. This will tell us what values map to `int8.min` and `int8.max`.

![Min and Max Calculation](/media/m.png)
*Figure 1: Visual representation of min and max calculation, W_max and W_min are the maximum and minimum values in tensor to be quantized, these values then map to int8.max and int8.min respectively.*


2) **Zero Point Calculation:**
    We can think of zero point as the point where the `float16` value of `0` lies in the `int8` data type. This basically maps the real number `r=0` to a quantized integer.


![Zero Point Calculation](/media/zp.png)
*Figure 2: Visual representation of zero point calculation. Z on the quantized axis is the zero point, and represents where the r=0.0 lies on the quantized axis.*

3) **Scale Calculation:**
    The scale basically tells us, how much each unit in the quantized data type represents in the original data type. Imagine a scale of `1.0`, this means that each unit in the quantized data type represents `1.0` in the original data type. The larger the scale, the larger is the original input range.


After these steps, we have everything we need to quantize and dequantize the data. With `r` being the real number, `q` being the quantized number, `Z` being the zero point, and `S` being the scale, the quantization and dequantization can be done by the following equations:

$$
q = \text{round}\left(\frac{r}{S}\right) + Z
$$
$$
r = (q - Z) \cdot S
$$

With some additional math, we can also derive the scale and zero point equations from the min and max values.

$$
S = \frac{W_{max} - W_{min}}{Q_{max} - Q_{min}} = \frac{W_{max} - W_{min}}{127 - (-128)}
$$
$$
Z = \text{round}\left(Q_{min} - \frac{W_{min}}{S}\right) = \text{round}\left(-128 - \frac{W_{min}}{S}\right)
$$

##  Implementation

```python
import torch
from collections import namedtuple

QTensor = namedtuple(
    "QTensor", ["tensor", "scale", "zero_point"]
)  # we need to track the scale and zero point to dequantize the tensor later


def quantize_tensor(tensor: torch.Tensor) -> QTensor:
    W_min = tensor.min()
    W_max = tensor.max()

    Q_min = torch.iinfo(torch.int8).min  # Get the minimum value of the int8 data type
    Q_max = torch.iinfo(torch.int8).max  # Get the maximum value of the int8 data type

    S = (W_max - W_min) / (Q_max - Q_min)  # Calculate the scale
    Z = torch.round(Q_min - (W_min / S))  # Calculate the zero point

    quantized_tensor = torch.round(tensor / S) + Z  # Quantize the tensor

    return QTensor(
        tensor=quantized_tensor.to(torch.int8), scale=S, zero_point=Z
    )  # Return the quantized tensor, scale, and zero point


def dequantize_tensor(q_tensor: QTensor) -> torch.Tensor:
    return (
        q_tensor.tensor.to(torch.float16) - q_tensor.zero_point
    ) * q_tensor.scale  # simply compute the real value from the already computed data
```

### Clipping
You might have noticed that we lose quite a lot of information while quantizing the tensor. This might lead to a precision loss, which can be detrimental to the performance of the model. Imagine a scenario where our input tensor distribution looks like the following:

![Input Distribution](/media/Outliers.png)
*Figure 3: A pretty common distribution of weights, where most of the values are centered around 0.0.*

Now imagine, we have a single data-point, which is far from the distribution, let's say `W_max=1000.0`. If we try to quantize this tensor, the scale would be very large, therefore a distance of `1` in the quantized data would represent a very large distance in the original unquantized data.
But remember, our input tensor is distributed around `0.0`, with most values lying in the range of `[-10.0, 10.0]`. This means that most of these values would be quantized to the same value, therefore losing a lot of information leading to a loss in performance.

To fix this issue, we can use a method called `Clipping`. This method involves clipping the values of the tensor to a certain range, and then quantizing the tensor. Our PyTorch implementation can be extended to include this method by the following:

```python
def quantize_tensor(
    tensor: torch.Tensor, clip_min: float | None = None, clip_max: float | None = None
) -> QTensor:
    if clip_min or clip_max: # check if atleast one of the clip values is provided
        tensor = torch.clamp(tensor, clip_min, clip_max)

    W_min = tensor.min()
    W_max = tensor.max()

    Q_min = torch.iinfo(torch.int8).min  # Get the minimum value of the int8 data type
    Q_max = torch.iinfo(torch.int8).max  # Get the maximum value of the int8 data type

    S = (W_max - W_min) / (Q_max - Q_min)  # Calculate the scale
    Z = torch.round(Q_min - (W_min / S))  # Calculate the zero point

    quantized_tensor = torch.round(tensor / S) + Z  # Quantize the tensor

    return QTensor(
        tensor=quantized_tensor.to(torch.int8), scale=S, zero_point=Z
    )  # Return the quantized tensor, scale, and zero point
```
This is the only required change to the implementation. In this implementation, we choose the clipping values manually, but in production cases, the clipping values are usually computed from the data distribution via different methods, such as `Percentile Clipping`, or even optimization methods such as minimizing the `KL Divergence` between the original and dequantized distribution. This process is called `Calibration`.

### Quantization Granularity

Another method to improve the performance of the quantized model is called `Quantization Granularity`. With the above implementation, we are computing the scale and zero point for the entire tensor. We could improve on this, by computing these values for a sub-part of the tensor. There are different variants of tensor splitting, such as `Per Channel`, `Per Token`, etc. The only difference between these variants is across which dimension is the zero point and scale computed. This can further lead to a better performance of the model, with cost of only a few extra bytes in memory. To save time and space, we will not be implementing these methods from scratch here, but just have this in mind when currently used quantization schemes are shown.

## Quantization Aware Training 

With this out of the way, we can finally take a look at the concept of `Quantization Aware Training`. To further improve the performance of the quantized model at inference time, we can use the concept of Quantization Aware Training. This method involves making the model *used to* quantized weights, activations respectively. This involves training or fine-tuning the model with something called `Fake Quantization`. This is a method to simulate the quantization process during training. This is accomplished by doing the following:

- Original weights are stored in the original data type, such as `float16`.
- Computation is done in the original data type.
- After we load the weights, we quantize them and then dequantize back. This is done to simulate the loading of integer weights and their dequantization done during the inference.
- The activations of the previous layer can be quantized, then dequantized back to get the values that the model would be using during the inference. This depends whether we're doing both activation and weight quantization, or only weight quantization.
- We then backpropagate the gradient w.r.t. the dequantized weights, and update the original weights.

You might be wondering, why do we compute with the dequantized weights, but update the original weights? If you remember our implementation, to dequantize the tensor, we round the values to the nearest integer. This means that if the gradient w.r.t. the dequantized weights is used to update the dequantized weights, the change could be too small to be further visible in the integer representation, therefore we would lose the update information. We can think of this as *passing* the gradient to the original weights. This is called `STE` or `Straight Through Estimator`. 

It's easier to see this in a diagram:

![Quantization Aware Training](/media/QAT.png)
*Figure 4: Visual representation of Quantization Aware Training. In this diagram, we can see that the weights are stored in the original data type, and are quantized and dequantized during the forward pass. This simulates the inference process, where the weights are loaded from memory and then dequantized. During the backward pass, we pass the gradient to the original weights, which are then updated.*

###  Implementation
To properly implement this, we would like to replace all of the `torch.nn.Linear` layers with our own custom implementation. This custom implementation would involve the following:

- Quantizing the weights
- Dequantizing the weights.
- Computing the forward pass with the dequantized weights.
- Backpropagating the error w.r.t. the dequantized weights.
- Updating the original weights with the gradient.

To do this, we can register a custom autograd function in PyTorch.

```python
class FakeQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        W_min, W_max = tensor.min(), tensor.max()
        Q_min, Q_max = torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max
        scale = (W_max - W_min) / (Q_max - Q_min)
        zero_point = torch.round(-Q_min - (W_min / scale))

        # Quantize
        quantized = torch.round(tensor / scale) + zero_point
        quantized = torch.clamp(quantized, Q_min, Q_max)

        # Dequantize
        dequantized = (quantized - zero_point) * scale

        # Save mask for backward pass
        mask = (quantized >= Q_min) & (quantized <= Q_max)
        ctx.save_for_backward(mask)

        return dequantized

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, 
        grad_output: torch.Tensor
    ) -> tuple[torch.Tensor]:
        (mask,) = ctx.saved_tensors
        return grad_output * mask
```

This function is used to simulate the quantization process during training. Note the `mask` variable, this is used to store a boolean mask 
of values that weren't clipped during the forward pass. Therefore, values that were clipped are not updated in the backward pass. This helps simulate the inference process and stabilizes the training process.


After this, we can create a `torch.nn.Module` that encapsulates the `FakeQuantizeFunction` and replace all of the `torch.nn.Linear` layers with this module. We can do this using a simple utility function.

```python
class QuantizedLinear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Quantize weights
        quantized_weight = FakeQuantizeFunction.apply(self.weight)

        # Use quantized weights for the linear operation
        return torch.nn.functional.linear(input, quantized_weight, self.bias)
```

```python
def replace_layers_with_quantized(model: torch.nn.Module) -> torch.nn.Module:
    """Replaces all Linear layers with QuantizedLinear layers"""
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, QuantizedLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None
            ))
            # Copy the weights and bias
            getattr(model, name).weight.data = module.weight.data
            if module.bias is not None:
                getattr(model, name).bias.data = module.bias.data
        else:
            replace_layers_with_quantized(module)
    return model
```

Now, we can create our own model and use this utility function to do `QAT`.

```python
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)

model = replace_layers_with_quantized(model)

# Now we can train the model as usual
# ...
```

## How to do this in practice?

As you might have noticed, we have written quite a lot of code, which doesn't handle a lot of edge cases and is rather simple. We can use PyTorch's [AO](https://github.com/pytorch/ao) library to do this for us. This library provides a [QAT](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat) module, which provides functionality for different quantization schemes.

The examples shown were purely educational. In production, different quantization schemes are used to improve the performance of the model. `AO` currently provides 2 different quantization schemes (note the different granularities of quantization, as discussed in the [Quantization Granularity](#quantization-granularity) section), which are:

1) **int8 per token dynamic activation quantization with int4 per group weight quantization:**
    This method quantizes weights to `int8` and activations to `int4`. Then, computation is done in original data type, that is `float16` usually. This is a good starting point for quantization aware training.

2) **int4 per group weight quantization:**
    This method quantizes weights to `int4`, but keeps the activations in `float16`. Then, weights are dequantized `on the fly` during the `matmul` kernel call. This is just to optimize the latency and performance of the model.


To reproduce our example with `AO`, you can use the following code:

```python
import torch
from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer

model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
).cuda()

# Quantizer for int8 dynamic per token activations +
# int4 grouped per channel weights, only for linear layers
qat_quantizer = Int8DynActInt4WeightQATQuantizer()

# Insert "fake quantize" operations into linear layers.
model = qat_quantizer.prepare(model)

# Now we can train the model as usual
# ...
```

Recalling our example, we have replaced the `torch.nn.Linear` layers with our own custom implementation. Equivalent of this is done by the `qat_quantizer.prepare(model)` method. This method inserts the `FakeQuantizeFunction` into the linear layers, and replaces the weights in the matrix multiplication with the quantized weights.

However, in case of inference, we do not want to do these steps. We only need to dequantize the weights and do the computation, as the weights are already quantized when loaded from memory. We haven't implemented this in our example, as its not relevant to the concept, but is required in production setting. `AO` provides a `qat_quantizer.convert(model)` method, which does this for us. We can use the following code to achieve this:

```python
# Convert the model to the quantized model
# This replaces all the "fake quantize" operations with the actual quantize operations
model = qat_quantizer.convert(model)

# Now we can use the model for inference
# ...
```

## Tips and Tricks

- **Finetuning:** finetuning the model with `QAT` is usually a better approach then training the model from scratch.
- **Layers:** quantizing only some layers is a good approach, some layers are influenced more by the quantization process. Try experimenting with replacing only some layers. Replacing the later layers is usually better than replacing the earlier layers. Also, in general it's not a good approach to quantize critical layers, such as attention. A good approach is to quantize the feed-forward layers, as those are the ones that require the most memory.

## Conclusion

In this tutorial, we have looked at the concept of quantization aware training, how does it work in depth, its benefits and how to implement it from scratch in PyTorch, and how to use `AO` to do this for us, which is a lot more efficient approach.








