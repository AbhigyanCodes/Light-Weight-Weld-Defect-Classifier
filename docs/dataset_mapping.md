# Mapping of numerical features

- crack_count: number of connected components identified as crack-like in edge map (area filtered).
- crack_length: sum of perimeter estimates of detected components.
- shear_length: estimated longest near-horizontal line length found by HoughLinesP.

These are heuristic features; for improved robustness consider morphological filtering, U-Net segmentation, and refined measurement techniques.
