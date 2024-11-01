import numpy as np
import matplotlib.pyplot as plt


# Planck's law equation
def blackbody_spectrum(wavelength, T=5778, normalise=False):
    # Constants
    h = 6.626e-34  # Planck's constant in J*s
    c = 3.0e8  # Speed of light in m/s
    k = 1.38e-23  # Boltzmann constant in J/K
    a = 2.0 * h * c**2
    b = h * c / (wavelength * k * T)
    intensity = a / (wavelength**5 * (np.exp(b) - 1))

    if normalise:
        intensity= intensity/np.max(intensity)

    return intensity


if __name__ == "__main__":

    # Temperature of the Sun in Kelvin
    T_sun = 5778  # approximate effective temperature of the Sun in Kelvin
    T_particles_phoebe = 70
    T_sat = 273 - 140
    T = T_sat

    # Wavelength range (in meters)
    wavelengths = np.arange(0.250e-6, 50e-6, 10e-9)  # from 1 nm to 3000 nm
    wavelengths = np.arange(0.25e-6, 40e-6, 10e-9)  # from 1 nm to 3000 nm

    # Calculate blackbody spectrum for the Sun
    spectrum = np.array(blackbody_spectrum(wavelengths, T, normalise=True))
    # print(spectrum)
    max_intens = np.max(spectrum)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths * 1e6, spectrum, color='black')
    plt.title(f'Blackbody Spectrum T={T}K')
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel('Intensity (W/m^2/nm)')
    plt.grid(True)
    plt.show()

    # Print the wavelength and intensity columns with a tab in between
    for wavelength, intensity in zip(wavelengths, spectrum):
        print(f"{wavelength * 1e9:.2f}\t{intensity/max_intens:.2f}")
