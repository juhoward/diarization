import torch, os, math
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load_inline
os.environ['CUDA_LAUNCH_BLOCKING']='1'

img = io.read_image('/home/digitalopt/Pictures/Webcam/cards2/card.jpg')

def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext")

cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''

cuda_src = cuda_begin + r'''
__global__ void rgb2grey_kernel(int blockidx, int threadidx, int blockdim, unsigned char* x, unsigned char* out, int n){
  int i = blockidx * blockdim + threadidx;
  if (i < n) {
    out[i] = 0.2989f * x[i] + 0.5870f * x[i + n] + 0.1140f * x[i + 2 * n];
  }
}

torch::Tensor rgb_to_grayscale(torch::Tensor input) {
  CHECK_INPUT(input);
  int h = input.size(1);
  int w = input.size(2);
  printf("h*w: %d*%d\n", h,w);
  auto output = torch::empty({h,w},input.options());
  int threads = 256;
  rgb2grey_kernel<<<cdiv(w*h, threads), threads, blockdim, x, out, n>>>(
    input.data_ptr<int>(), output.data_ptr<int>(), w*h, x, out, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
'''

cpp_src = "torch::Tensor rgb_to_grayscale(torch::Tensor input);"
module = load_cuda(cuda_src, cpp_src, ['rgb_to_grayscale'], verbose=True)