# User-provided data manifests

Each nonempty line is a JSON object. Image and label paths may be relative to the manifest's containing directory. `id` must be unique. `source_id` identifies the original clean source across all variants and splits; do not split variants from one source across train/validation/test.

Example only (no image or annotation is bundled):

```json
{"id":"train/source001/variantA","source_id":"source001","dataset":"INRIA","attack":"AdvP","image":"../images/patched001.png","clean":"../images/clean001.png","patch_boxes_xyxy":[[100,90,150,140]],"person_boxes_xyxy":[[70,20,210,400]],"person_iscrowd":[0],"person_gt_available":true}
```

All inline boxes use **original-image pixel xyxy**, not416-grid coordinates. `clean` must have the same original dimensions as `image`; Stage-2 reconstruction training requires it. Stage-1 patch training requires patch boxes, Stage-2 person training requires complete person boxes. AP evaluation includes every original object, including people not selected as attack targets.

Instead of inline boxes, `patch_labels` and `person_labels` may refer to YOLO txt files with `class cx cy width height`, normalized to original dimensions. Source class1 means patch; source class0 means person. Both become the positive detector class1 internally. Preserve original crowd flags where available; do not silently clip amodal boxes or invent person GT from patch boxes.

Fill separate JSONL files at `manifests/stage1_train.jsonl`, `stage1_validation.jsonl`, `stage2_train.jsonl`, `stage2_validation.jsonl`, and locked test manifests, using your own permitted assets. These names match the example configs. The source ids and complete raw annotation provenance must be verified by the dataset preparer. Evaluation with clean comparisons requires clean references on every record; partial paired coverage raises an error.

The two training corpora differ. `scripts/prepare_dataset.py --stage stage1` accepts original patch-only records without a clean reference or person GT; this preserves usable AdvTs localization assets. `--stage stage2` requires complete original person GT and paired clean training/validation images and does not require a GT patch mask. Both stages can write into one manifest directory while checking cross-stage source split consistency; existing manifest files are never overwritten. Default `--stage both` is only for records legitimately having both annotations and clean references; do not use it to invent missing labels or force the two corpora to have identical attack families.
