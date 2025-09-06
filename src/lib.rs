#[cfg(test)] use burn::backend::{LibTorch, NdArray, Wgpu};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{LeakyRelu, LeakyReluConfig, PaddingConfig2d};
use burn::prelude::*;


/// Construct a standard 3x3 Conv2dConfig
/// 
/// [RRDBNet] uses pretty much the same 3x3 convolution layer throughout the whole
/// network, only varying the number of input and output features.
fn std_conv2d(in_f: usize, out_f: usize) -> Conv2dConfig {
    Conv2dConfig::new([in_f, out_f], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1))
}


/// Configuration for [SimplifiedBlock].
#[derive(Config, Copy, Debug)]
pub struct SimplifiedBlockConfig {
    /// Number of feature channels
    pub nf: usize,
}

impl SimplifiedBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SimplifiedBlock<B> {
        let &SimplifiedBlockConfig { nf } = self;
        SimplifiedBlock {
            conv1: std_conv2d(nf, nf).init(device),
            lrelu: LeakyReluConfig::new().with_negative_slope(0.2).init()
        }
    }
}


#[derive(Debug, Module)]
pub struct SimplifiedBlock<B: Backend> {
    pub conv1: Conv2d<B>,
    pub lrelu: LeakyRelu
}

impl<B: Backend> SimplifiedBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // The problem happens with the combination of LeakyRelu and Conv2d. If
        // I use either one alone, the test passes.
        self.lrelu.forward(self.conv1.forward(x))
    }
}


#[cfg(test)]
fn compare_backends<Reference: Backend, Tested: Backend>() {
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::{Distribution, cast::ToElement};

    // be reproducible
    Reference::seed(0xd14bccffe1928888);

    const NF: usize = 3;
    let config = SimplifiedBlockConfig::new(NF);
    let ref_device = <Reference as Backend>::Device::default();
    let ref_module = config.init::<Reference>(&ref_device);
    let test_device = <Tested as Backend>::Device::default();
    let test_module = config.init::<Tested>(&test_device);

    // copy weights from ref_module to test_module
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let bytes = recorder.record(ref_module.clone().into_record(), ()).unwrap();
    let record = recorder.load::<SimplifiedBlockRecord<Tested>>(bytes, &test_device).unwrap();
    let test_module = test_module.load_record(record);

    // generate random input and run it through ref_module
    let ref_input = Tensor::<Reference, 4>::random([2, NF, 2, 2], Distribution::Uniform(-1.0, 1.0), &ref_device);
    let ref_output = ref_module.forward(ref_input.clone());

    // copy input to the test backend and run it through test_module
    let test_input = Tensor::<Tested, 4>::from_data(ref_input.to_data(), &test_device);
    let test_output = test_module.forward(test_input);

    // prepare tensors for comparison on the reference backend
    let expected = ref_output;
    let actual = Tensor::<Reference, 4>::from_data(test_output.to_data(), &ref_device);


    println!("expected: {:?}", expected);
    println!("actual: {:?}", actual);

    let close = expected.is_close(actual, None, None);
    println!("is_close: {:?}", close);

    assert!(close.all().into_scalar().to_bool());
}

#[test]
fn test_libtorch() {
    compare_backends::<NdArray, LibTorch>();
}

#[test]
fn test_wgpu() {
    compare_backends::<NdArray, Wgpu>();
}