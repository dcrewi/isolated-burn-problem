use burn::backend::{LibTorch, NdArray, Wgpu};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{LeakyRelu, LeakyReluConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::Distribution;


/// Construct a standard 3x3 Conv2dConfig
/// 
/// [RRDBNet] uses pretty much the same 3x3 convolution layer throughout the whole
/// network, only varying the number of input and output features.
fn std_conv2d(in_f: usize, out_f: usize) -> Conv2dConfig {
    Conv2dConfig::new([in_f, out_f], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1))
}


/// Configuration for [ResidualDenseBlock].
#[derive(Config, Copy, Debug)]
pub struct ResidualDenseBlockConfig {
    /// Number of feature channels
    pub nf: usize,
    /// Growth channel, ie additional intermediate channels added during the
    /// densely connected layers
    pub gc: usize,
}

impl ResidualDenseBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualDenseBlock<B> {
        let &ResidualDenseBlockConfig { nf, gc } = self;
        ResidualDenseBlock {
            conv1: std_conv2d(nf, gc).init(device),
            conv2: std_conv2d(nf + 1 * gc, gc).init(device),
            conv3: std_conv2d(nf + 2 * gc, gc).init(device),
            conv4: std_conv2d(nf + 3 * gc, gc).init(device),
            conv5: std_conv2d(nf + 4 * gc, nf).init(device),
            lrelu: LeakyReluConfig::new().with_negative_slope(0.2).init()
        }
    }
}


/// Residual dense block with five convolution layers.
/// 
/// See [Densely Connected Convolutional Networks (arXiv:1608.06993)](https://arxiv.org/abs/1608.06993)
#[derive(Debug, Module)]
pub struct ResidualDenseBlock<B: Backend> {
    pub conv1: Conv2d<B>,
    pub conv2: Conv2d<B>,
    pub conv3: Conv2d<B>,
    pub conv4: Conv2d<B>,
    pub conv5: Conv2d<B>,
    pub lrelu: LeakyRelu
}

impl<B: Backend> ResidualDenseBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x0 = x;
        let x1 = self.lrelu.forward(self.conv1.forward(x0.clone()));
        let x0x1 = Tensor::cat(vec![x0.clone(), x1], 1);
        let x2 = self.lrelu.forward(self.conv2.forward(x0x1.clone()));
        let x0x1x2 = Tensor::cat(vec![x0x1, x2], 1);
        let x3 = self.lrelu.forward(self.conv3.forward(x0x1x2.clone()));
        let x0x1x2x3 = Tensor::cat(vec![x0x1x2, x3], 1);
        let x4 = self.lrelu.forward(self.conv4.forward(x0x1x2x3.clone()));
        let x0x1x2x3x4 = Tensor::cat(vec![x0x1x2x3, x4], 1);
        let x5 = self.conv5.forward(x0x1x2x3x4);
        x5 * 0.2 + x0
    }
}


fn compare_backends<Reference: Backend, Tested: Backend>() {
    let config = ResidualDenseBlockConfig::new(4, 4);
    let ref_device = <Reference as Backend>::Device::default();
    let ref_module = config.init::<Reference>(&ref_device);
    let test_device = <Tested as Backend>::Device::default();
    let test_module = config.init::<Tested>(&test_device);

    // copy weights from ref_module to test_module
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let bytes = recorder.record(ref_module.clone().into_record(), ()).unwrap();
    let record = recorder.load::<ResidualDenseBlockRecord<Tested>>(bytes, &test_device).unwrap();
    let test_module = test_module.load_record(record);

    // generate random input and run it through ref_module
    let ref_input = Tensor::<Reference, 4>::random([2, 4, 9, 9], Distribution::Uniform(0.0, 1.0), &ref_device);
    let ref_output = ref_module.forward(ref_input.clone());

    // copy input to the test backend and run it through test_module
    let test_input = Tensor::<Tested, 4>::from_data(ref_input.to_data(), &test_device);
    let test_output = test_module.forward(test_input);

    // prepare tensors for comparison on the reference backend
    let expected = ref_output;
    let actual = Tensor::<Reference, 4>::from_data(test_output.to_data(), &ref_device);

    assert!(actual.all_close(expected, None, None));
}

#[test]
fn test_libtorch() {
    compare_backends::<NdArray, LibTorch>();
}

#[test]
fn test_wgpu() {
    compare_backends::<NdArray, Wgpu>();
}