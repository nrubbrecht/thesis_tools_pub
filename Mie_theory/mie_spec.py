import miepython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from blackbody_spectrum import blackbody_spectrum

# Mie theory simulations of VIMS spectra

def select_lambdas(array, targets):
    df = pd.DataFrame()
    if isinstance(targets, float):
        # Calculate the absolute differences between each row's first entry and the target
        differences = (array["lambda"] - targets).abs()
        # Find the index of the row with the minimum absolute difference
        closest_row_index = differences.idxmin()
        # Return the row with the minimum absolute difference
        row = array.iloc[[closest_row_index]]
        # print(row)
        # df = df.append(row)
        df = pd.concat([df, row])
    else:
        for target in targets:
            # Calculate the absolute differences between each row's first entry and the target
            differences = (array["lambda"] - target).abs()
            # Find the index of the row with the minimum absolute difference
            closest_row_index = differences.idxmin()
            # Return the row with the minimum absolute difference
            row = array.iloc[[closest_row_index]]
            # print(row)
            # df = df.append(row)
            df = pd.concat([df, row])

    return df


def extract_optical_constants(file_path):
    """
    Extracts the wavelengths, real part of refractive index,
    and imaginary part of refractive index for crystalline H2O ice at 130 K
    from the given text file.

    Parameters:
    file_path: str - Path to the text file.

    Returns:
    pd.DataFrame - DataFrame containing 'lambda', 'n', and 'k' columns.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize data list
    data = []

    # Read the file line by line
    for line in lines:
        stripped_line = line.strip()

        # Skip header lines and empty lines
        if not stripped_line or "---" in stripped_line or "Units" in stripped_line:
            continue

        # Split line into columns
        columns = stripped_line.split()

        # Check if there are enough columns
        if len(columns) >= 5:
            # Check for crystalline form and temperature
            if columns[0] == "Crystalline" and columns[1] == "130":
                # Extract the wavelength, n, and k values
                try:
                    wavelength = float(columns[2])  # Wavelength in microns
                    n = float(columns[3])  # Real part of refractive index
                    k = float(columns[4])  # Imaginary part of refractive index
                    data.append((wavelength, n, k))
                except ValueError as e:
                    print(f"Skipping line due to ValueError: {stripped_line} - {e}")

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data, columns=['lambda', 'n', 'k'])
    return df

def mie_spectrum(wavelengths, radius, angle_deg):
    ice_data_warren = pd.read_csv('../opt_cons/warren_2008_ASCIItable.txt', sep='\s+', header=None)
    ice_data_warren.columns = ['lambda', 'n', 'k']  # wavelength in microns

    # Set angle for intensity calculation
    theta = np.deg2rad(angle_deg)
    mu = np.cos(theta)

    # Calculate blackbody spectrum for the Sun
    sun_wavelengths = np.arange(250e-9, 5051e-9, 1e-10)  # from 1 nm to 3000 nm
    sun_spectrum = np.array(blackbody_spectrum(sun_wavelengths))
    sun_max = np.max(sun_spectrum)

    # lets call this phase function for ease in not changing every name (however, it is not the same as phase function)


    intensity_vs_wavelength = []

    for wavelength in wavelengths:
        # Calculate the size parameter
        size_parameter = 2 * np.pi * radius / wavelength

        # Get refractive index for ice at this wavelength
        df_ice = select_lambdas(ice_data_warren, [wavelength])

        n_ice = df_ice['n'].iloc[0]
        k_ice = df_ice['k'].iloc[0]

        m = complex(float(n_ice), float(k_ice))  # Complex refractive index of ice

        # Calculate unpolarized intensity at this angle
        intensity = miepython.i_unpolarized(m, size_parameter, mu, norm="one")
        intensity = intensity * blackbody_spectrum(wavelength * 10 ** -6) / sun_max

        intensity_vs_wavelength.append(intensity)

    return np.array(intensity_vs_wavelength)


def mie_spectrum_normal(wavelengths, mean_radius, std_radius, angle_deg, num_samples=100):
    # Read the ice data from the Warren 2008 dataset
    ice_data_warren = pd.read_csv('../opt_cons/warren_2008_ASCIItable.txt', sep='\s+', header=None)
    ice_data_warren.columns = ['lambda', 'n', 'k']  # wavelength in microns

    # Set angle for intensity calculation
    theta = np.deg2rad(angle_deg)
    mu = np.cos(theta)

    intensity_vs_wavelength = []

    for wavelength in wavelengths:
        # Get refractive index for ice at this wavelength
        df_ice = select_lambdas(ice_data_warren, [wavelength])
        n_ice = df_ice['n'].iloc[0]
        k_ice = df_ice['k'].iloc[0]
        m = complex(float(n_ice), float(k_ice))  # Complex refractive index of ice

        # Generate particle sizes based on normal distribution
        radii = np.random.normal(mean_radius, std_radius, num_samples)
        radii = radii[radii > 0]  # Filter out negative values
        intensities = []

        for radius in radii:
            # Calculate the size parameter for each particle
            size_parameter = 2 * np.pi * radius / wavelength
            # Calculate unpolarized intensity at this angle
            intensity = miepython.i_unpolarized(m, size_parameter, mu, norm="Bohren")
            intensities.append(intensity)

        # Average the intensity over the particle size distribution
        mean_intensity = np.mean(intensities)
        intensity_vs_wavelength.append(mean_intensity)

    return np.array(intensity_vs_wavelength)


def scattering_intensity_unpolarized(wavelengths, radius, refractive_index_medium, n_particle_real, k_particle_imag, I0,
                                     distance, theta):
    """
    Calculate the scattered intensity for unpolarized light using Mie theory.

    Parameters:
    wavelengths: Array of wavelengths (in microns).
    radius: Radius of the scattering sphere (in microns).
    refractive_index_medium: Refractive index of the surrounding medium (dimensionless).
    n_particle_real: Array of real part of the refractive index of the particle (same size as wavelengths).
    k_particle_imag: Array of imaginary part of the refractive index of the particle (same size as wavelengths).
    I0: Incident light intensity (arbitrary units).
    distance: Distance from the scattering sphere (in microns).
    theta: Scattering angle in degrees (0° = forward scattering, 180° = backscattering).

    Returns:
    intensity_vs_wavelength: Array of scattered intensity as a function of wavelength.
    """

    # Convert theta to radians
    theta_rad = np.deg2rad(theta)

    # Initialize an empty list to store the calculated intensities
    intensity_vs_wavelength = []

    for i, wavelength in enumerate(wavelengths):
        # Size parameter (x = 2 * pi * radius / wavelength)
        size_parameter = 2 * np.pi * radius / wavelength

        # Refractive index of the particle (complex number)
        m = complex(n_particle_real[i], k_particle_imag[i])

        # Calculate the amplitude scattering functions for S1 and S2 (perpendicular and parallel polarizations)
        S1, S2 = miepython.mie_S1_S2(m, size_parameter, theta_rad, norm="Bohren")

        # Unpolarized light: average of the squared magnitudes of S1 and S2
        amplitude_scattering_function = (np.abs(S1) ** 2 + np.abs(S2) ** 2) / 2

        # k = 2 * pi * refractive_index_medium / wavelength
        k = 2 * np.pi * refractive_index_medium / wavelength

        # Calculate the scattered intensity based on the unpolarized light formula
        intensity = (I0 * amplitude_scattering_function) / (k ** 2 * distance ** 2)

        # Append the result to the intensity array
        intensity_vs_wavelength.append(intensity)

    return np.array(intensity_vs_wavelength)



if __name__ == "__main__":

    # wavelengths = np.arange(1.2, 5, 0.01)  # from 400 nm to 1000 nm (in microns)
    # df = extract_optical_constants("Mastrapa_nk.txt")
    #
    # # df_ice = select_lambdas(df, wavelengths)
    # df_ice = df
    # # print(df)
    # # print(df_ice)
    # # exit()
    # n_ice = df_ice['n'].values
    # k_ice = df_ice['k'].values
    # wavelengths = df_ice['lambda'].values
    # # Parameters
    # radius = 4  # Radius of the sphere (in microns)
    # refractive_index_medium = 1.0  # Refractive index of the surrounding medium (vacuum = 1.0)
    # I0 = 1.0  # Incident light intensity
    # distance = 1e6  # Distance from the scattering sphere (in microns)
    # theta = 20  # Scattering angle in degrees
    #
    # # Calculate the scattered intensity for unpolarized light
    # intensity_vs_wavelength = scattering_intensity_unpolarized(wavelengths, radius, refractive_index_medium,
    #                                                            n_ice, k_ice, I0, distance, theta)
    # # Plot the spectrum
    # plt.figure(figsize=(8, 6))
    # plt.plot(wavelengths, intensity_vs_wavelength, label=f'Scattering angle = {theta}°')
    # plt.xlabel("Wavelength (microns)")
    # plt.ylabel("Scattered Intensity (arbitrary units)")
    # plt.title("Unpolarized Scattered Intensity Spectrum using Mie Theory")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # exit()

    # Example parameters
    radius = 1 # in microns
    wavelengths = np.arange(0.8, 5, 0.01)  # from 400 nm to 1000 nm (in microns)
    angle_deg = 20  # Scattering angle in degrees

    angle_degs = [0, 20, 160 , 180]
    for angle in angle_degs:
        # Get the intensity as a function of wavelength
        intensity_spectrum = mie_spectrum(wavelengths, radius, angle)
        plt.plot(wavelengths, intensity_spectrum, label=f"{angle}")

    # Plot intensity vs wavelength
    # plt.plot(wavelengths, intensity_spectrum)
    plt.title(f'Intensity vs Wavelength at {angle_deg} degrees')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Intensity')
    plt.yscale('log')  # Log scale for better visualization of scattering intensity
    plt.grid(True)
    plt.legend()
    plt.show()

    sizes = [1, 1.5, 2, 2.5, 3]
    std_size = 0.5
    angle = 20
    for size in sizes:
        # Get the intensity as a function of wavelength
        intensity_spectrum = mie_spectrum(wavelengths, size, angle)
        # intensity_spectrum = mie_spectrum_normal(wavelengths, size, std_size, angle, num_samples=50)
        plt.plot(wavelengths, intensity_spectrum, label=f"{size}")

        # Plot intensity vs wavelength
        # plt.plot(wavelengths, intensity_spectrum)
    plt.title(f'Intensity vs Wavelength at {angle} degrees')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Intensity')
    plt.yscale('log')  # Log scale for better visualization of scattering intensity
    plt.grid(True)
    plt.legend()
    plt.show()


    ice_data_warren = pd.read_csv('../opt_cons/warren_2008_ASCIItable.txt', sep='\s+', header=None)
    ice_data_warren.columns = ['lambda', 'n', 'k']  # wavelength in microns
    df_ice = select_lambdas(ice_data_warren, wavelengths)
    n_ice = df_ice['n']
    k_ice = df_ice['k']
    print(n_ice)
    #
    # plt.plot(wavelengths, n_ice, label="n real")
    # plt.plot(wavelengths, k_ice, label="k imaginary")
    # plt.legend()
    # plt.show()
