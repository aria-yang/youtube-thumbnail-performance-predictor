# SHAP Interpretation Notes

## How to interpret SHAP values
- A positive SHAP value means the feature pushed the model toward a higher logit for a class on that example.
- A negative SHAP value means the feature pushed the model away from that class on that example.
- Larger absolute SHAP values mean the feature had more influence on the prediction.
- Global importance is computed as mean absolute SHAP value, so it tells us which features mattered most overall, not whether they helped or hurt on average.

## Patterns that would support our thesis
- OCR features ranking near the top would support the claim that thumbnail text design contributes meaningfully to engagement prediction.
- Face features such as `num_faces`, `largest_face_area_ratio`, or emotion indicators ranking highly would support the claim that human presence and expressed emotion matter.
- If both OCR and face features appear in the global top features rather than only CNN embedding dimensions, that supports the multimodal thesis more strongly than a vision-only story.
- If the strongest non-CNN features are intuitive, such as more text density, numeric text, or expressive faces, that strengthens the interpretability argument for the project.
- If OCR and face features have negligible SHAP importance compared with CNN embeddings, that would weaken the thesis that these handcrafted multimodal cues add meaningful explanatory value.
