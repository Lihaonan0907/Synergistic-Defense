# Implementation notes

The paper does not specify every implementation detail. This code uses explicit
choices for the spectral input interface, high-pass enhancement receptive field,
repair network widths and layers, wavelet boundaries, outer loss scales,
training phase boundaries, and data/evaluation setup. These choices are defined
in the configurations and source; exact equivalence to an unavailable original
implementation is not claimed.

- Wavelet processing uses four-level symmetric db6 decomposition with learned
  encoders. The high-pass enhancement is used during training in `paper_literal`
  mode.
- The localizer uses torchvision's torch ROIAlign backend for higher-order
  derivatives. The AFAL outer scale and enhancement learning-rate multiplier are
  configuration choices where the paper does not specify these values.
- The repair network concatenates spatial and frequency features to predict a
  masked residual. Network widths and layer topology are implementation choices.
- The person detector uses a separately obtained initializer. Training consumes
  patched training images and paired clean images as specified by the dataset
  manifest, and uses the phase boundaries specified by the configuration.
