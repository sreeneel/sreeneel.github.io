---
layout: post
title:  "Metal(MPS backend) support for Upsample3d in Pytorch"
date:   2025-02-05 08:04:08 -0800
categories: jekyll update
permalink: /upsample3d_mps
custom_css: upsample3d_mps
css: upsample3d_mps
---

* toc
{:toc}

<br>
<hr>
__PR for "Pytorch MPS Upsample3D Nearest Neighbor"__ : [Pytorch MPS Upsample3D Nearest Neighbor Patch][pytorch-mps-patch]

__And here's the branch__: [mps_upsample3d][mps_upsample3d-branch]

<b>Note<sup>1</sup></b>: <i>A lot of the functions in above code are named with suffix <b>`mps`</b> as pytorch internally named a lot of these functionx as such. `MPS(Metal Performance Shaders)` is a higher level framework that provides data-parallel primitives so that apps can utlizie Metal hardware "without needing to create and maintain hand-written shaders for each GPU family" [Metal Performance Shaders Documentation][MPS-Documentation]</i>

This post demonstrates use of the underlying `metal` framework directly and writing a custom `Compute Kernel` to perform 3D Upsampling
<hr>
<br>

A while ago, I was trying to execute a 3d diffusion model on my M1 Macbook Pro (M1 Max Apple Silicon). Given the documentation here "[Accelerated PyTorch training on Mac][apple-pytorch-mps]" and "[MPS Backend][pytorch-mps-backend]", 
I was hoping that I could leverage all the compute and memory of my M1 Macbook Pro (10 Core CPU, 32 Core GPU, dedicated ANE & 64GB of Unified Memory). 

During the run, I started seeing a bunch of warnings on the console something like:

```console
UserWarning: The operator 'aten::upsample_nearest3d.vec' is not currently 
supported on the MPS backend and will fall back to run on the CPU.
This may have performance implications.(Triggered internally at 
/Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/mps/MPSFallback.mm:14.)
  return torch._C._nn.upsample_nearest3d(input, output_size, scale_factors)
```

This seemed like a fun exercise to grok through `Pytorch` codebase and learn more about its internals and how to add a new native operator support.

`torch.nn.Upsample` upsamples a tensor either by a `scale_factor` or to a given target output size.

In the rest of this post, I will walk through the steps I took to add support for upsample_nearest_3d operator to pytorch.

__Here's the entire PR__ : [Pytorch MPS Upsample3D Nearest Neighbor Patch][pytorch-mps-patch]

__And here's the branch__: [mps_upsample3d][mps_upsample3d-branch]



## 1. Setup a `dev` environment

* Fork and clone pytorch repo from here: [pytorch github][pytorch-github]
* Create a `Conda` environment
```zsh
conda create -n pytdevel_py312  python=3.12
conda activate pytdevel_py312
```
* Build the repo using following commands in your terminal app
    ```zsh
    cd <root-of-cloned-pytorch-git-repo>
    # Following 3 steps below are needed if you need to make a clean build
    #1 git submodule deinit -f .
    #2 git clean -xdf
    #3 python setup.py clean
    git submodule update --init --recursive
    DEBUG=1 python setup.py develop
    ```
    When installing pytorch with `python setup.py develop`, an `.egg-link` is created in `site-packages` folder

## 2. Setup a test case for `upsample3d` operator
All the tests are located in `test` directory under the root of pytorch git project.
Looking through `test/test_mps.py`, I found `test_upsample_nearest_2d` function, which was easy enough to modify to adapt to test `upsample_nearest3d`. 

Here's the new test-case: [test/test_mps.py::test_upsample_nearest3d()][test_upsample_neares3d_diff]

Above test-function 
* Creates an input-tensor of a given shape with `requires_grad=True`
* Moves this tensor to MPS
* Performs `torch.nn.Upsample` on both CPU and MPS backends. 
* Also performs the backward pass on both tensors.
* Compares the results of Forward & Backward Pass results from both backends
 
Given that `MPS` is falling-back to `CPU`, outputs should match to begin with.

Here's how to run this test
```zsh
cd <root-of-cloned-pytorch-git-repo>
python test/test_mps.py TestMPS.test_upsample_nearest3d

```
## 3. Remove default CPU Fallback behavior
Currently `upsample_nearest3d.vec` is made to fallback to CPU as there's no Metal implementation.

As per the warning message above, this seems to be in `MPSFallback.mm`
`(Triggered internally at 
/Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/mps/MPSFallback.mm:14.)
  return torch._C._nn.upsample_nearest3d(input, output_size, scale_factors)`

We need to remove the following line in `aten/src/ATen/mps/MPSFallback.mm` , so that pytorch will start to use the new native metal functions we will implement shortly

```objc++
...<snip>...
 m.impl("upsample_nearest3d.vec", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
 }
```

## 4. Register the op in `aten/src/ATen/native/native_functions.yaml`

Register a new (mps)backend for `upsample_nearest3d` in `aten/src/ATen/native/native_functions.yaml` as following:

forward function  : `upsample_nearest3d.out`

backward function : `upsample_nearest3d_backward`

```yml
- func: upsample_nearest3d.out(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
  python_module: nn
  structured: True
  dispatch:
    CPU: upsample_nearest3d_out_cpu
    CUDA: upsample_nearest3d_out_cuda
    MPS: upsample_nearest3d_out_mps

... <snip> ...

- func: upsample_nearest3d_backward.grad_input(Tensor grad_output, SymInt[3] output_size, SymInt[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
  python_module: nn
  structured: True
  dispatch:
    CPU: upsample_nearest3d_backward_out_cpu
    CUDA: upsample_nearest3d_backward_out_cuda
    MPS: upsample_nearest3d_backward_out_mps
```

relevant patch: [native_functions.yaml][native_functions_yaml_patch]

## 5. Locate where implementation needs to go
Looking at the `native_functions.yaml`, looks like `Upsample 2D` versions are named:
forward function: `upsample_nearest2d_out_mps`
backward function: `upsample_nearest2d_backward_out_mps`

Grepping for these symbols, we can see that these are defined in `aten/src/ATen/native/mps/operations/UpSample.mm`

## 6. Function definition

Following is function definition for `upsample_nearest3d_out_mps`:

[upsample_nearest3d_out_mps Function definition][nearest3d_out_mps-func-def]
```c++
TORCH_IMPL_FUNC(upsample_nearest3d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 c10::optional<double> scales_d,
 c10::optional<double> scales_h,
 c10::optional<double> scales_w,
 const Tensor& output) {
    upsample_nearest3d_out_mps_impl(input, output_size, scales_d, scales_h, scales_w, output);
}
```

Looking at the signature, this function takes:
* `input`       : const reference to `input` Tensor
* `output_size` : required target size
* `scales_d`    : An optional double on how much to scale on `depth` dim
* `scales_h`    : An optional double on how much to scale on `height` dim
* `scales_w`    : An optional double on how much to scale on `width` dim
* `output`      : const ref to `output` Tensor

`Tensor` is a central pytorch light-weight, templated, wrapper class that holds a shared/ref-counted `TensorImpl` object. 
`Tensor` class can be thought of as an `intrusive pointer` where it manages life-time of `TensorImpl` object.
`TensorImpl` is the actual low-level class containing contiguous storage & its metadata. It is the "target" of the "intrusive_ptr"(i.e `Tensor` class). 

`IntArrayRef` is a pytorch templated class that represents a constant-reference to an array of contiguously laid out elements
This class does __NOT own__ underlying memory. This class is primarily used to pass around __consecutive elements__ easily between functions and APIs, for eg. to pass `output_sizes` in the above API

This function calls a helper function `upsample_nearest3d_out_mps_impl` to do the heavy-lifting

## 7. Metal Setup

This is the main driver for setting up MetalComputePipeline, threadgroups and dispatching threads to execute the upsample compute kernel.
Here's a broad outline of the flow:

* Get shared `MPSDevice`(Pytorch's RAII wrapper managing Metal's `MTLDevice`) : [Code][Get-MTLDevice]
* Get `MTLComputeCommandEncoder` (using Pytorch's wrapper `MPSStream`) : [Code][Get-MTLComputeCommandEncoder]
* Create `MTLComputePipelineState` using helper function `upsample3DNearestNeighborPSO` : [Code][Get-MTLComputePipelineState]

  ### 7.1 MetalPipelineStateObject  Setup : 
  [mps::upsample3DNearestNeighborPSO][upsample3DNearestNeighborPSO-Code]
    * Lookup local Cache and return if a `MTLPipelineState` object exists for key=`upsample_nearest3d_kernel`: [Code][PSO-Cache-Lookup]. 
      * `psoCache` stores `id<MTLComputePipelineState>` for Values, which being an obj-c pointer  type, will be default constructed to `nil` in objc++. Next line checks to see if it is non-nil and if so returns cached object. Otherwise it will proceed to create a new PSO which is an expensive operation
    * Compile Upsample3D Kernel : [Call CompileKernel][Call compileUpsample3dKernelLibrary] / [compileUpsample3dKernelLibrary-Implementation][compileUpsample3dKernelLibrary Impl]
    * Extract forward compute kernel `upsample_nearest3d_kernel` function from this library. [Extract Upsample Forward function][Extract Upsample Forward function]
    * create new Compute Pipeline State with the forward function [Create PipelineStateObject with Upsample forward-fn][PSO-Creation]
    * Store the newly created object in PipelineStateObject Cache Before returning the newly created object

        <i><b>Note<sup>1</sup></b>: Setting up MTLComputePipelineState object is an expensive operation as the Compute-kernel needs to be compiled from a string, on the fly. This is the reason that it is cached once created.</i>

        <i><b>Note<sup>2</sup></b>: `TORCH_CHECK` macro helps in error-checking of various function calls such as `newFunctionWithName:` and `newComputePipelineStateWithFunction:error:`</i>

  ### 7.2 Configure "3D" ThreadGroup for dispatching work to Metal
  [Threadgroup Config Code][thrgrp-config-code]
  
  Few notes about how Metal (or any GPU) executes a kernel
  * Metal executes a given kernel function over a <b> 1D/2D/3D</b> `grid of threads`.
  * Each "point" in the grid represents a single execution instance of the kernel function. Each `Thread` is executed on a single <b><i>Execution Unit (=~ FP ALU) </i></b>
  * Threads are further organized into <b>1D/2D/3D `threadgroups`. </b>
    * All threads in a threadgroup are executed together and share common GPU memory (such as L1 cache), but will have its own register-file
    * Threads in a threadgroup can collaborate if necessary to work on a sub-problem ie if kernel is written in such a way to collaborate
  * Proper configuration of `Dispatch grid` is essential to ensure the compute-task is not under-utilizing the GPU

  So given that out Upsample3d's tensors are of <b><i>N x C x D<sub>in</sub> x H<sub>in</sub> x W<sub>in</sub> </i></b>, I have organized all the required threads into :
  * <b><u>ThreadGroups</u></b> : Organized into a `2D` grid
  * <b><u>Threads</u></b>      : Each of the above ThreadGroups is organized into a `3D` grid of threads

[![](assets/img/3DUpsampleMetalThreadDispatchGrid_Small.png){: .thumbnail}](assets/img/3DUpsampleMetalThreadDispatchGrid.png)
  
  #### 7.2.1 ThreadGroup grid Calculation
    Metal MTLComputePipelineState exposes two methods to help figure out number of threads per threadgroup:
    * `threadExecutionWidth`          : Number of threads GPU can schedule in parallel
    * `maxTotalThreadsPerThreadgroup` : Max # of threads that can be in a single threadgroup
    <br><br>
    So for a given PipelineState, we can launch maximum possible threadgroups using following code: 
    ```objc
    NSUInteger w = kernelPSO.threadExecutionWidth; 
    NSUInteger h = kernelPSO.maxTotalThreadsPerThreadgroup / w;
    NSUInteger d = 1;
    MTLSize bdim = MTLSizeMake(w, h, d);
    ```
    This sets up a 2D grid of threadgroups with 
    <br>
    w=`kernelPSO.threadExecutionWidth` & 
    <br>
    h=`kernelPSO.maxTotalThreadsPerThreadgroup/w`

  #### 7.2.2 Threads within a ThreadGroup grid Calculation
  As per the diagram above, threads within a threadgroup are arranged in a 3D-grid.
  <br>Dimensions of this grid `gW x gH x gD` are calculated as:
    ```c++
    auto outNumEl = output.numel();
    auto gW = (outWidth + w - 1) / w;
    auto gH = (outHeight + h - 1) / h;
    auto gD = (outNumEl + (gH * gW) - 1) / (gH * gW);
    MTLSize tgpg = MTLSizeMake(gW, gH, gD);
    ```
  <i><b>Note<sup>1</sup></b> : Once we fix `w` & `h`, we need to ensure that every pixle is assigned to a thread. Consequently these calculations depend on both `output.numel()` as well as dimensions of `threadgroup 2D grid`
  <br><i><b>Note<sup>2</sup></b> : `outWidth` = `output.size(4)` i.e size of 4th dim of output.size(<b><i>N x C x D<sub>in</sub> x H<sub>in</sub> x W<sub>in</sub></i></b>), which happens to be the <u>width</u> of the output tensor </i>

  #### 7.2.3 Launch the kernel with Dispatch-Grid
  ```objc++
  [computeEncoder dispatchThreadgroups:tgpg threadsPerThreadgroup:bdim];
  ```

## 8. Metal Kernel Implementation

[Link to Kernel Code][upsample_nearest3d_kernel]

Upsample essentially makes a tensor "bigger" by up-sizing its dimensions either by a given scale-factor or a given target dimension shape.
There are multiple strategies as to how it fills up the new data in each dimension such as `nearest neighbor`, `linear`, `bilinear`, `bicubic`, `trilinear`

For this post/implementation, we will be looking at `nearest` neighbor strategy.

More concretely here's an example from pytorch documentation:
```python
>>> input
tensor([[[[1., 2.],
          [3., 4.]]]])
>>> m = nn.Upsample(scale_factor=2, mode='nearest')
>>> m(input)
tensor([[[[1., 1., 2., 2.],
          [1., 1., 2., 2.],
          [3., 3., 4., 4.],
          [3., 3., 4., 4.]]]])
```
Documentation here: [Pytorch Upsample Documentation][pytorch-upsample-doc]

#### 8.1 Calculating threadId for each Kernel
```c++
kernel void upsample_nearest3d_kernel(
    device float const* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant Upsample3DParams& upsample3DParams [[buffer(2)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {
    
    auto threadId = thread_position_in_grid.x +
        thread_position_in_grid.y * threads_per_grid.x +
        thread_position_in_grid.z * threads_per_grid.y * threads_per_grid.x;
    if(threadId > upsample3DParams.dstNumEl) {
        return;
    }
    _upsample_nearest3d_kernel_impl(src, dst, upsample3DParams, threadId);
}
```
[Link to Code][threadId-Calc]

When a kernel is submitted for execution, it executes over a 2D grid of threadgroups, where each threadgroup is a 3D grid of threads. 

A `thread` is an instance of the kernel-function that executes for each point in this grid.
`thread_position_in_grid` is a 3-dimensional vector identifying each thread's position in the above grid-of-grids.

Since our src & dst data are contiguously laid out 1D memory location, We need to translate this 3-d positional vector into 1D indices into src & dst memory locations.

so in the above formula, threadId is calculated as :

<i>Num steps on the last dim: </i> : `thread_position_in_grid.x`
<br><i>Multiply with `y's stride`</i> : `thread_position_in_grid.y * threads_per_grid.x`
<br><i>Multiply with `z's stride`</i> : `thread_position_in_grid.z * threads_per_grid.y * threads_per_grid.x`

Sometimes Metal(most GPUs) can dispatch <u>more threads</u> than that can service your data, for <u>alignment reasons</u>. So we need to make sure we check that calculated threadId is within the bounds of the data
```
if(threadId > upsample3DParams.dstNumEl) {
  return;
}
```

#### 8.2 Calculating `dstIndex` and `srcIndex` from threadId

`upsample(nearest)` operation requires us to copy/replicate data from `nearest neighbors` in the `src` buffer into `dst` buffer. Something like:

dst[<span style="color:#c357cc; font-size:14px; font-family: Menlo;">dstIndex</span>] = src[<span style="color:#11bf11; font-size:14px; font-family: Menlo;">srcIndex</span>]

This operation needs to be performed for each <u>data-point in the `dst` buffer</u>. In this implementation, this op is expressed as a compute kernel. 

As this kernel will be executed parallely, we will need to figure out correct `src-index` and `dst-index` for each data-point in the tensor.

`threadId` calculated above gives a linear-index of the thread executing the kernel.

<span style="color:#c357cc; font-size:14px; font-family: Menlo;">dstIndex</span>: we can use `threadId` as an index into `dst` buffer directly.

<span style="color:#11bf11; font-size:14px; font-family: Menlo;">srcIndex</span>: We need to translate/convert this <span style="color:#c357cc; font-size:14px; font-family: Menlo;">dstIndex(=~threadId)</span> to figure out an index into `src` buffer from where to copy the data from.

<b>Calculating srcIndex</b>

Grid parameters are passed into the kernel using a custom struct `Upsample3DParams`. This struct contains widths of each dimension i.e `N, C, D, H, W`

Using these params, we can calculate strides in each dimension for <span style="color:#c357cc; font-size:14px; font-family: Menlo;">dst-tensor-strides</span> as follows:
```c++
size_t dst_h_stride = upsample3DParams.dstWidth;
size_t dst_d_stride = upsample3DParams.dstHeight 
                        * upsample3DParams.dstWidth;
size_t dst_c_stride = upsample3DParams.dstDepth 
                        * upsample3DParams.dstHeight
                        * upsample3DParams.dstWidth;
size_t dst_n_stride = upsample3DParams.numChannels
                        * upsample3DParams.dstDepth 
                        * upsample3DParams.dstHeight
                        * upsample3DParams.dstWidth;
```
[Link to Code][dst_strides_calc]

We can now use these strides to calculate dst-tensor-dims indices for the given threadId <span style="color:#c357cc; font-size:14px; font-family: Menlo;">dstDIndx, dstHIndx, dstWIndx</span> as follows:
```c++
size_t batchSz = threadIdx / dst_n_stride; //same for src & dst
threadIdx = threadIdx % dst_n_stride;
size_t chSz = threadIdx / dst_c_stride; //same for src & dst
threadIdx = threadIdx % dst_c_stride;
size_t dstDIndx = threadIdx / dst_d_stride;
threadIdx = threadIdx % dst_d_stride;
size_t dstHIndx = threadIdx / dst_h_stride;
threadIdx = threadIdx % dst_h_stride;
size_t dstWIndx = threadIdx;
```
[Link to Code][dst-tensor-dims-indices]

Note that `BatchSize` and `ChannelSize` will not change during Upscaling.

Once we have <span style="color:#c357cc; font-size:14px; font-family: Menlo;">dst-dim-indices for curr threadId</span>, scale-factors for each dim, <span style="color:#11bf11; font-size:14px; font-family: Menlo;">src-dim-indices</span>
<br>we can calculate <span style="color:#11bf11; font-size:14px; font-family: Menlo;">srcDIndx, srcHIndx, srcWIndx</span> as follows <i>(showing for `depth` here, same applies for other dims)</i>

<b>srcD = min(floor(dstD * scaleD), srcD-1)</b>
<br>

```c++
auto srcDIndx = nn_compute_source_index(upsample3DParams.depthScale,
                    dstDIndx, upsample3DParams.srcDepth);
auto srcHIndx = nn_compute_source_index(upsample3DParams.heightScale,
                    dstHIndx, upsample3DParams.srcHeight);
auto srcWIndx = nn_compute_source_index(upsample3DParams.widthScale,
                    dstWIndx, upsample3DParams.srcWidth);

//and nn_compute_source_index is:
int nn_compute_source_index(const float scale, int dst_index, int input_size) {
  const int src_index = min(static_cast<int>(floor(dst_index * scale)),
                              input_size - 1);
  return src_index;
}
```
[[src-tensor-dims-indices][src-tensor-dims-indices]][[nn_compute_source_index][nn_compute_source_index]]

Once we have <span style="color:#11bf11; font-size:14px; font-family: Menlo;">srcIndx, srcHIndx, srcWIndx</span>, and previously calculated srcStrides, we can calculate linear <b><u>srcIndex</u></b> as :

```c++
auto srcIdx = (batchSz * src_n_stride) + 
                (chSz * src_c_stride) +
                (srcDIndx * src_d_stride) + 
                (srcHIndx * src_h_stride) + srcWIndx;

//and then finally copy value from src buffer to dst-buffer
dst[dstIdx] = src[srcIdx];
```
[Link to Code][srcIdx-Calculation]

## 9. Misc Tips
#### 9.1 Enable Debug logging for Metal Kernels

<i>Note: Works only with Metal V3.2+</i>

* Set Compiler options to include `MTLLanguageVersion3_2`:

  ```c++
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion3_2];
  options.enableLogging = true;
  NSString* upsampleKernelStr = [NSString stringWithCString:UPSAMPLE_NEAREST3D_KERNEL
      encoding:NSASCIIStringEncoding]
  upsampleMtlLibrary = [device newLibraryWithSource: upsampleKernelStr
                            options:options
                            error:&error];
  ```

* In the <b><u>metal kernel</u></b>

  ```
  #include <metal_logging>
  constant metal::os_log custom_log("com.my-sub-sys", "my-sub-cat");

  //at log-site, say if you want to print 3d-coords of thread in the grid:
  custom_log.log("thread_position_in_grid: %v3hlu", thread_position_in_grid);
  ```

* Make logging not redact "<i>private</i>" data in console log

  Launch your process with `OS_ACTIVITY_DT_MODE=Enable-Private-Data` env-var

  Optionally you can also set log-level with env-var : and `MTL_LOG_LEVEL=MTLLogLevelDebug`

  ```
  MTL_LOG_LEVEL=MTLLogLevelDebug OS_ACTIVITY_DT_MODE=Enable-Private-Data python test/test_mps.py TestMPS.test_upsample_nearest3d
  ```

  <b><i>Note</i></b> Make sure that you dont log so much that the underlying buffer overflows causing dropped messages

#### 9.2 Launching into debugger

  <b><i>lldb</i></b> can be launched normally as follows:

  ```zsh
>> lldb python
(lldb) target create "python"
Current executable set to '/Users/sreeneel/miniconda3/envs/pytdevel_py312/bin/python' (arm64).
(lldb) b  b upsample_nearest3d_out_mps_impl
Breakpoint 1: no locations (pending).
(lldb) r test/test_mps.py TestMPS.test_upsample_nearest3d
Process 50320 launched: '/Users/sreeneel/miniconda3/envs/pytdevel_py312/bin/python' (arm64)
1 location added to breakpoint 1
* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 5.1
  frame #0: 0x00000003079683d0 libtorch_cpu.dylib`at::native::upsample_nearest3d_out_mps_impl(input=0x000000016fdf9600, output_size=at::IntArrayRef @ 0x000000016fdf7140, scales_d= Has Value=false , scales_h= Has Value=false , scales_w= Has Value=false , output=0x000000016fdf7328) at UpSample.mm:771:7
  ```

  <b><i>Note</i></b> Make sure that executable is set to the right conda env in which you built `pytorch`
  
  you can also look at the callstack to understand how execution flows upto this point




[apple-pytorch-mps]: https://developer.apple.com/metal/pytorch/
[pytorch-mps-backend]: https://pytorch.org/docs/main/notes/mps.html
[MPS-Documentation]: https://developer.apple.com/documentation/metalperformanceshaders
[pytorch-upsample-doc]: https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
[pytorch-mps-patch]: https://github.com/sreeneel/pytorch/pull/5/files
[mps_upsample3d-branch]: https://github.com/sreeneel/pytorch/tree/mtl_upsample3d
[pytorch-github]: https://github.com/pytorch/pytorch
[test_upsample_neares3d_diff]: https://github.com/sreeneel/pytorch/pull/5/files#diff-fdee6c1352305cfc875c3f14dc43440e2c681f8f229ef8977289bb7fbe389966R6520-R6567
[native_functions_yaml_patch]: https://github.com/sreeneel/pytorch/pull/5/files#diff-2f3dbd85efb9b5172f2264eedd3be47dd765e6ab7cc8bf3ade5e62c28ae35991R12910-R12937
[nearest3d_out_mps-func-def]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R916-R924
[Get-MTLDevice]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R759
[Get-MTLComputeCommandEncoder]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R764
[Get-MTLComputePipelineState]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R769
[upsample3DNearestNeighborPSO-Code]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R245-R262
[Call compileUpsample3dKernelLibrary]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R254
[compileUpsample3dKernelLibrary Impl]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R226-R243
[PSO-Cache-Lookup]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R248
[Extract Upsample Forward function]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R255
[PSO-Creation]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R257
[thrgrp-config-code]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R780-R798
[upsample_nearest3d_kernel]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R112-R126
[threadId-Calc]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R118-R123
[dst_strides_calc]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R83-R86
[dst-tensor-dims-indices]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R93-R101
[src-tensor-dims-indices]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R103-R105
[nn_compute_source_index]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R54-R57
[srcIdx-Calculation]: https://github.com/sreeneel/pytorch/pull/5/files#diff-5a9e9fba28930af9dd45fd5b9bb47d85145252d6a265a45000f14bb3327fd8d9R106-R108
