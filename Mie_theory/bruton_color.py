import numpy as np
import matplotlib.pyplot as plt

# Function to convert wavelength to pre-gamma corrected RGB
def wl2rgb_pregamma(wl):
    if 380 <= wl < 440:
        s = 0.3 + (0.7 * (wl - 380.0)) / (420.0 - 380.0) if wl < 420 else 1.0
        r = (s * -1 * (wl - 440)) / (440 - 380)
        g = 0.0
        b = s
    elif 440 <= wl < 490:
        r = 0.0
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wl < 510:
        r = 0.0
        g = 1.0
        b = (-1 * (wl - 510)) / (510 - 490)
    elif 510 <= wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= wl < 645:
        r = 1.0
        g = (-1 * (wl - 645)) / (645 - 580)
        b = 0.0
    elif wl >= 700:
        r = 0.3 + (0.7 * (780 - wl)) / (780 - 700)
        g = 0.0
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    return [r, g, b]


# Function to apply gamma correction to the RGB values
def wl2rgb(wl, gamma=0.8):
    r, g, b = wl2rgb_pregamma(wl)
    return [np.power(r, gamma), np.power(g, gamma), np.power(b, gamma)]


def adjust_color_intensity(base_rgb, intensity):
    """
    Adjusts the saturation of the base color based on the normalised intensity (0 white to 1 saturation).
    Higher intensity keeps the color close to its base value, lower intensity makes it closer to white.
    """
    # Intensity adjustment: linearly interpolate between white and base color
    adjusted_color = [1 - (1 - c) * intensity for c in base_rgb]

    return adjusted_color


if __name__ == "__main__":

    # Create a canvas to draw the spectrum
    fig, ax = plt.subplots(figsize=(10, 2))  # Width of 10 and height of 2 for a long, thin canvas

    # Define the range of wavelengths (in nm)
    wavelengths = np.arange(380, 780)

    # Draw the spectrum
    for wl in wavelengths:
        r, g, b = wl2rgb(wl)
        ax.add_patch(plt.Rectangle((wl - 380, 0), 1, 50, color=(r, g, b), transform=ax.transData))

    # Set axis limits and remove axis labels and ticks
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 50)
    ax.axis('off')  # Hide the axes

    # Display the plot
    plt.show()
