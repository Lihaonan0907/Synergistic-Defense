# Implementation notes

The paper does not specify every implementation detail. This code uses explicit
choices for the spectral input interface, high-pass enhancement receptive field,
repair network widths and layers, wavelet boundaries,
training phase boundaries, and data/evaluation setup. These choices are defined
in the configurations and source; exact equivalence to an unavailable original
implementation is not claimed.

- The patch localizer uses four-level symmetric db2 decomposition; the repair
  branch uses four-level symmetric db6 decomposition. Both use learned encoders.
  The high-pass enhancement is used during training in `paper_literal` mode.
- The localizer uses torchvision's torch ROIAlign backend for higher-order
  derivatives. The AFAL global alignment coefficient and sub-band base weight
  are each 10.0; the enhancement learning-rate multiplier is an implementation
  choice where the paper does not specify a value.
- The repair network concatenates spatial and frequency features to predict a
  masked residual. Network widths and layer topology are implementation choices.
- The person detector uses a separately obtained initializer. Training consumes
  patched training images and paired clean images as specified by the dataset
  manifest, and uses the phase boundaries specified by the configuration.

The current configuration was aligned to the paper without retraining or
re-evaluating models. The paper specifies Python 3.8 and PyTorch 1.13.1, whereas
this repository's supported environment uses Python 3.10/3.11 and PyTorch 2.1;
equivalence across environments and results has not been established.
