import numpy as np

from preprocessing import preprocess_image


# Create a fake grayscale image
image = np.random.randint(
    0,
    255,
    (256, 256),
    dtype=np.uint8
)


# Preprocess
processed = preprocess_image(
    image
)


print("Original shape:", image.shape)

print(
    "Processed shape:",
    processed.shape
)

print(
    "Minimum value:",
    processed.min()
)

print(
    "Maximum value:",
    processed.max()
)


# Validation
assert processed.shape == (128, 128)

assert processed.min() >= 0

assert processed.max() <= 1


print("\nPreprocessing test PASSED!")