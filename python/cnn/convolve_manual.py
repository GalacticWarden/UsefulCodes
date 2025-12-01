def convolve(image_array, kernel, padding='same'): # both image_array, kernel as numpy array 
    """
    Applies a 2D convolution filter to an image.

    Args:
        image_array (np.array): The input image as a NumPy array (height, width, channels).
        kernel (np.array): The convolution kernel (filter) as a NumPy array.
        padding (str): 'same' for output size same as input, 'valid' for no padding.

    Returns:
        np.array: The filtered image.
    """
    if image_array.ndim == 2: # Handle grayscale images
        image_array = np.expand_dims(image_array, axis=-1)

    image_height, image_width, num_channels = image_array.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding if 'same'
    if padding == 'same':
        pad_h = kernel_height // 2
        pad_w = kernel_width // 2
        padded_image = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
    elif padding == 'valid':
        padded_image = image_array
        pad_h, pad_w = 0, 0
    else:
        raise ValueError("Padding must be 'same' or 'valid'")

    output_height = image_height - kernel_height + 1 + 2 * pad_h
    output_width = image_width - kernel_width + 1 + 2 * pad_w
    output_array = np.zeros((output_height, output_width, num_channels), dtype=image_array.dtype)

    for c in range(num_channels):
        for i in tqdm(range(output_height)):
            for j in range(output_width):
                # Extract the window
                window = padded_image[i:i + kernel_height, j:j + kernel_width, c]
                # Perform element-wise multiplication and sum
                output_array[i, j, c] = np.sum(window * kernel)

    return output_array
