import cv2
import numpy as np

def darken_image(image_path, output_path, alpha=0.5, beta=0):
    """
    Darken an image by adjusting its brightness and contrast.

    :param image_path: Path to the input image.
    :param output_path: Path where the output image will be saved.
    :param alpha: Contrast control (1.0-3.0). Less than 1.0 will reduce contrast.
    :param beta: Brightness control (0-100). Negative values will darken the image.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Apply the transformation
    darker_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Save the result
    cv2.imwrite(output_path, darker_image)

    # Display the original and darkened images
    cv2.imshow("Original Image", image)
    cv2.imshow("Darker Image", darker_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
darken_image("aeroplane0.jpeg", "aeroplane0-darker.jpeg", alpha=0.5, beta=-50)
