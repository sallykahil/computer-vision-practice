# Computer Vision Labs

This project contains Jupyter notebook labs for learning computer vision techniques using OpenCV, NumPy, and Matplotlib.

## Architecture
- **Labs**: Individual Jupyter notebooks in `Lab1/` covering topics like image processing, representation, and morphological operations
- **Data**: Images and datasets stored in `data/` directory
- **Environment**: Python virtual environment in `myvenv/` with required packages

## Key Imports and Conventions
Always use these import aliases:
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
```

## Image Handling Patterns
- **Loading**: `image = cv.imread("../data/filename.jpg")` (relative path from Lab1/)
- **Displaying**: `cv.imshow("Window Title", image); cv.waitKey(0)`
- **Saving**: `cv.imwrite("output.jpg", image)`
- **Color spaces**: Convert with `cv.cvtColor(image, cv.COLOR_BGR2GRAY)` or `cv.COLOR_BGR2RGB`
- **Channels**: Split with `(r, g, b) = cv.split(rgb_image)`

## Geometric Transformations
- **Translation**: `cv.warpAffine(image, np.float32([[1,0,x],[0,1,y]]), (w, h))`
- **Rotation**: `cv.getRotationMatrix2D(center, angle, scale)` then `cv.warpAffine`

## Drawing and Canvas Creation
- Create blank canvas: `canvas = np.zeros((height, width, 3), dtype="uint8")`
- Draw shapes: `cv.line(canvas, pt1, pt2, color, thickness)`
- Text: `cv.putText(canvas, text, org, font, scale, color, thickness)`

## Morphological Operations
- Define kernel: `kernel = np.ones((5,5), np.uint8)`
- Erosion: `cv.erode(image, kernel, iterations=1)`
- Dilation: `cv.dilate(image, kernel, iterations=1)`

## Development Workflow
- Work in Jupyter notebooks for interactive experimentation
- Images display in separate windows via `cv.imshow`
- Use `cv.waitKey(0)` to pause execution until key press
- Clean up windows with `cv.destroyAllWindows()` when needed

## File Organization
- Notebooks reference data with `../data/` paths
- Output images saved in current directory or specified paths
- Bank note data in `data/bank_note_data.txt` for classification tasks