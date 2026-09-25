# Innovation Notes

This file is reserved for the innovation investigation and method comparison.
It is deliberately separated from the reproducible execution package.

Current implementation choices:

- Text representation: BERT `bert-base-uncased`, last hidden state, 50 tokens.
- Audio representation: openSMILE eGeMAPSv02 low-level descriptors, 25 dimensions.
- Visual representation: OpenFace 17 action units and 3 head-pose angles.
- Alignment: 0.5 second segments, 50 positions, zero padding.

Possible later extension:

- Add a second audio or visual extractor and compare feature quality under the
  same validation protocol.
- Add word-level timestamps for a stricter text-to-time mapping study.
- Separate invalid openSMILE sentinel values from valid feature statistics.

The final paper should cite only methods actually used and must not introduce
external emotion datasets prohibited by the competition rules.

