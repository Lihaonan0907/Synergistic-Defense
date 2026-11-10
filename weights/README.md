# Model weights

Model weights are not included. Inference and evaluation require compatible
Stage-1 and Stage-2 exported bundles and adjacent `<weight>.json` sidecars.
The sidecar must contain a `sha256` field matching the weight file; the loader
verifies this hash and loads tensors strictly. The supplied default sidecars
describe the hashes of the corresponding expected bundles. For other compatible
bundles, use their own matching sidecars. Stage-1 must match the frozen
localizer in Stage-2.

Native training checkpoints are not inference bundles. This repository does
not currently include a native-checkpoint export utility.
