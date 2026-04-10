# Regression SHAP Interpretation Notes

## How to interpret SHAP values
- A positive SHAP value means the feature pushed the predicted regression score upward on that example.
- A negative SHAP value means the feature pushed the predicted regression score downward on that example.
- Larger absolute SHAP values mean the feature had more influence on the prediction.
- Global importance is computed as mean absolute SHAP value, so it shows which features matter most overall.

## Patterns that would support our thesis
- OCR features ranking near the top would support the claim that thumbnail text design contributes to predicted performance.
- Face features such as `num_faces`, `largest_face_area_ratio`, or emotion indicators ranking highly would support the claim that human presence and expression matter.
- If both OCR and face features appear near the top rather than only CNN embedding dimensions, that supports the multimodal thesis more strongly than a vision-only explanation.
- If OCR and face features have very small SHAP importance compared with CNN embeddings, that weakens the claim that these multimodal cues add meaningful explanatory value.
