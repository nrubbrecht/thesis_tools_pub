import miepython
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import scipy as sc
import scipy.special as scp

# sys.path.append('..\Thesis-main')
from Mie_theory.bruton_color import wl2rgb, adjust_color_intensity
from Mie_theory.blackbody_spectrum import blackbody_spectrum

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


def find_peaks(series):
    peaks, _ = sc.signal.find_peaks(series)
    return peaks


def mie_phase_func(wavelengths, radius, start_deg, end_deg, spacing, plot=False ,peaks=False, norm_type="albedo",
                   color_intensity=False, material_data_link='../opt_cons/warren_2008_ASCIItable.txt'):
    # Warren 2008: n-ice UV to IR
    # ice_data_warren = pd.read_csv('Thesis-main/opt_cons/warren_2008_ASCIItable.txt', sep='\s+', header=None)
    ice_data_warren = pd.read_csv(material_data_link, sep='\s+', header=None)

    ice_data_warren.columns = ['lambda', 'n', 'k']  # wavelength in micron

    # Angular range for plotting
    spacing = np.deg2rad(spacing)
    theta = np.arange(np.deg2rad(start_deg), np.deg2rad(end_deg)+spacing, spacing)
    mu = np.cos(theta)
    scat_angle = np.rad2deg(theta)

    # create an array to add all spectra
    phase_func_array = np.zeros((1, len(mu)))
    i = 0
    plt.style.use('dark_background')

    if wavelengths == "vis":
        wavelengths = np.linspace(0.38, 0.7, 11)
    elif wavelengths == "bl1":
        wavelengths = np.linspace(0.4, 0.5, 11)
    elif wavelengths == "red":
        wavelengths = np.linspace(0.55, 0.75, 11)
    # difficulties with assigning color to infrared because black does not blend
    elif wavelengths == "ir1":
        wavelengths = np.linspace(0.65, 0.85, 11)
    elif wavelengths == "ir3":
        wavelengths= np.linspace(0.875, 1.0, 11)

    # Calculate blackbody spectrum for the Sun
    sun_wavelengths = np.arange(250e-9, 5051e-9, 1e-10)  # from 1 nm to 3000 nm
    sun_spectrum = np.array(blackbody_spectrum(sun_wavelengths))
    sun_max =np.max(sun_spectrum)

    for wavelength in wavelengths:
        # Calculate the size parameter
        size_parameter = 2 * np.pi * radius / wavelength

        # select material data for specific wavelengths
        df_ice = select_lambdas(ice_data_warren, wavelength)
        lam_ice = float(df_ice['lambda'].iloc[0])
        n_ice = float(df_ice['n'].iloc[0])
        k_ice = float(df_ice['k'].iloc[0])

        refractive_index_particle = complex(n_ice, k_ice)  # Complex refractive index of the particle
        refractive_index_medium = 1.0            # Refractive index of the surrounding medium

        # Calculate the Mie coefficients
        m = refractive_index_particle / refractive_index_medium
        x = size_parameter

        # S1 S2 approach
        # scattering_coeffs = miepython.mie_S1_S2(m, x, mu, norm="bohren")
        # # Extract S1 and S2 from scattering_coeffs
        # S1 = scattering_coeffs[0]
        # S2 = scattering_coeffs[1]
        # phase_function = (abs(S1)**2 + abs(S2)**2) / 2

        # other approach get unpolarised intensity directly
        # intensity at each angle in the array mu.  Units [1/sr]

        intensity = miepython.i_unpolarized(m, x, mu, norm=norm_type)
        # lets call this phase function for ease in not changing every name (however, it is not the same as phase function)
        phase_function = intensity * blackbody_spectrum(wavelength*10**-6)/sun_max
        # print(wavelength,blackbody_spectrum(wavelength*10**-6)/sun_max)
        # save phase function to array
        phase_func_array = np.vstack((phase_func_array,phase_function))

        # get colors from wavelength
        wl_nano = wavelength*1000
        r, g, b = wl2rgb(wl_nano)
        i += 1
        if plot:
            plt.plot(scat_angle, phase_function, label=f'{round(wavelength,3)}', c=(r,g,b))

        if peaks:
            peak_indices = find_peaks(np.log10(phase_function))
            peak_angles = scat_angle[peak_indices]
            peak_spacing =  np.diff(peak_angles)
            print("wavelength=",wavelength)
            print("peak angles=", peak_angles)
            print("peak spacing =", peak_spacing)

        # plt.plot(np.rad2deg(theta), phase_function, label='Phase Function')

    # delete zero row at top
    phase_func_array = np.delete(phase_func_array, 0, 0)

    if plot:
        # Plot the phase function
        plt.title('Mie Scattering Phase Function')
        plt.yscale("log")
        plt.grid(alpha=0.5)
        plt.legend()
        plt.show()

    # calculate average intensity for each theta
    phase_avg = np.mean(phase_func_array, 0)
    # calculate prominent wavelength for each theta based of intensities of discrete wavelengths
    phase_func_array_norm = phase_func_array/np.sum(phase_func_array, 0)
    prom_lambdas = []
    prom_colors = np.zeros((1, 3))
    rgb_colors = np.array([wl2rgb(wl*1000) for wl in wavelengths])

    for i in range(len(theta)):
        # assign weight to relative intensity of every lambda at specific theta
        weights = phase_func_array_norm[:, i]
        prom_wavelength = np.sum(weights * wavelengths)
        prom_lambdas.append(prom_wavelength)
        # get colors by summing weights color values for each wavelength
        # prom_rgb = wl2rgb(prom_wavelength*1000)
        prom_rgb = np.dot(weights, rgb_colors)
        prom_colors = np.vstack((prom_colors,prom_rgb))
    prom_colors = np.delete(prom_colors, 0, 0)

    mie_theta = scat_angle  # Scattering angles for the scatter plot
    mie_color = prom_colors  # Colors for the scatter plot points

    if color_intensity:
        mie_intens_normalized = phase_avg / np.max(phase_avg)
        intensities = np.log10(phase_avg + 1e-10)  # Adding a small value to avoid log(0)
        # Normalize the intensities to [0, 1]
        mie_intens_normalized = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))

        for i, intensity in enumerate(mie_intens_normalized):
            mie_color[i] = adjust_color_intensity(mie_color[i], intensity)

    # plt.figure()
    # plt.plot(scat_angle, prom_lambdas)
    # plt.show()
    # Plot the combined color for each scattering angle
    # for i, angle in enumerate(scat_angle):
    #     plt.plot([angle, angle], [0, 1], color=combined_colors[i], linewidth=4)
    if plot:
        # plt.scatter(scat_angle, phase_avg, c=prom_colors, edgecolors='none')
        # plt.yscale("log")
        # plt.ylabel("Intensity")
        # plt.xlabel("Scattering angle [deg]")
        # plt.show()

        # Create a figure with a GridSpec layout
        fig = plt.figure(figsize=(10, 6))  # Adjusted height for better layout

        # Create a GridSpec with 2 rows and 1 column
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.1)

        # First plot (Top) - Color lines based on scattering angles
        ax1 = fig.add_subplot(gs[0, 0])
        for i, angle in enumerate(mie_theta):
            ax1.plot([angle, angle], [0, 1], color=mie_color[i], linewidth=4)

        # Hide y-axis for ax1
        # ax1.set_xticks([])
        # ax1.set_xticklabels([])
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        ax1.set_title('Resulting Colors Based on Scattering Angle')
        ax1.grid()

        # Second plot (Bottom) - Scatter plot with log y-scale
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Share x-axis with ax1
        sc = ax2.scatter(scat_angle, phase_avg, c=prom_colors, edgecolors='none')
        ax2.set_yscale("log")
        ax2.set_ylabel("Intensity")
        ax2.set_xlabel("Scattering Angle (degrees)")
        ax2.grid()

        # Show the plot with a tight layout
        # plt.tight_layout()
        plt.xlim(start_deg, end_deg)
        plt.show()


    # Reset to the default style
    plt.style.use('default')

    return scat_angle, phase_avg, prom_colors


def mie_phase_func_band(wavelengths, radius, start_deg, end_deg, spacing, colours=None, plot=True,peaks=False):
    # Warren 2008: n-ice UV to IR
    # ice_data_warren = pd.read_csv('isis/warren_2008_ASCIItable.txt', sep='\s+', header=None)
    ice_data_warren = pd.read_csv('../opt_cons/warren_2008_ASCIItable.txt', sep='\s+', header=None)
    ice_data_warren.columns = ['lambda', 'n', 'k']  # wavelength in micron

    # Angular range for plotting
    theta = np.linspace(np.deg2rad(start_deg), np.deg2rad(end_deg), int(180 / spacing))
    mu = np.cos(theta)
    scat_angle = np.rad2deg(theta)

    # create an array to add all spectra
    phase_sum = np.zeros((1, len(mu)))
    i = 0
    for wavelength in wavelengths:
        # Calculate the size parameter
        size_parameter = 2 * np.pi * radius / wavelength

        # select material data for specific wavelengths
        df_ice = select_lambdas(ice_data_warren, wavelength)
        lam_ice = float(df_ice['lambda'].iloc[0])
        n_ice = float(df_ice['n'].iloc[0])
        k_ice = float(df_ice['k'].iloc[0])

        refractive_index_particle = complex(n_ice, k_ice)  # Complex refractive index of the particle
        refractive_index_medium = 1.0            # Refractive index of the surrounding medium

        # Calculate the Mie coefficients
        m = refractive_index_particle / refractive_index_medium
        x = size_parameter
        scattering_coeffs = miepython.mie_S1_S2(m, x, mu)

        # # Extract S1 and S2 from scattering_coeffs
        # S1 = scattering_coeffs[0]
        # S2 = scattering_coeffs[1]
        # phase_function = (abs(S1)**2 + abs(S2)**2) / 2
        # log_phase_function = np.log10(phase_function)

        # other approach get unpolarised intensity directly
        # intensity at each angle in the array mu.  Units [1/sr]
        intensity = miepython.i_unpolarized(m, x, mu, norm="albedo")
        # lets call this phase function for ease in not changing every name (however, it is not the same as phase function)
        phase_function = intensity

        # add phase function to sum
        phase_sum += phase_function

        if plot:
            if colours == None:
                plt.plot(scat_angle, phase_function, label=f'{round(wavelength, 3)}')
            else:
                plt.plot(scat_angle, phase_function, label=f'{round(wavelength,3)}', c=colours[i])
        if peaks:
            peak_indices = find_peaks(phase_function)
            peak_angles = scat_angle[peak_indices]
            peak_spacing =  np.diff(peak_angles)
            print(peak_angles)
            print(peak_spacing)

        # plt.plot(np.rad2deg(theta), phase_function, label='Phase Function')
        i += 1
    if plot:
        # Step 4: Plot the phase function
        plt.title('Mie Scattering Phase Function')
        plt.yscale('log')
        plt.legend()
        plt.show()

    phase_sum = phase_sum/len(wavelengths)
    if plot:
        plt.plot(scat_angle, phase_sum.T)
        plt.ylabel("Intensity")
        plt.xlabel("Scattering angle [deg]")
        plt.yscale('log')
        plt.show()
    return scat_angle, phase_sum


def fraunhofer(particle_radius, wavelength, start_deg, end_deg):
    """

    Parameters
    ----------
    particle_radius in micron
    wavelength in micron

    Returns
    -------

    """
    scat_space = np.linspace(start_deg, end_deg, 1000) * np.pi / 180  # scattering angle range
    wavelength = wavelength * 1e-6
    r = particle_radius*1e-6  # particle radius
    x = list(2 * np.pi * r / wavelength)  # size parameter
    wavelength = list(wavelength)
    plt.figure()
    for j in range(len(x)):
        bes_argument = x[j] * np.sin(scat_space)
        j1 = scp.jv(1, bes_argument)  # Using scipy.special.jv for Bessel function of the first kind
        io_corona = (x[j] * (1 + np.cos(scat_space)) / 2 * j1 / bes_argument) ** 2
        # Plot the Bessel function
        plt.plot(scat_space * 180 / np.pi, io_corona, label=f"{int(wavelength[j] * 1e9)}nm")

    # plt.yscale("log")
    plt.show()
    return


def fraunhofer2(particle_radii, wavelengths, start_deg, end_deg):
    """
    Parameters
    ----------
    particle_radii : list of floats
        Particle radii in microns.
    wavelengths : list of floats
        Wavelengths in microns.
    start_deg : float
        Starting angle in degrees.
    end_deg : float
        Ending angle in degrees.
    """
    scat_space = np.linspace(start_deg, end_deg, 1000) * np.pi / 180  # scattering angle range

    plt.figure()

    for wavelength in wavelengths:
        combined_io_corona = np.zeros_like(scat_space)
        for r in particle_radii:
            r_m = r * 1e-6  # convert to meters
            wavelength_m = wavelength * 1e-6  # convert to meters
            x = 2 * np.pi * r_m / wavelength_m  # size parameter
            bes_argument = x * np.sin(scat_space)
            j1 = scp.jv(1, bes_argument)  # Bessel function of the first kind
            io_corona = (x * (1 + np.cos(scat_space)) / 2 * j1 / bes_argument) ** 2
            combined_io_corona += io_corona

        plt.plot(scat_space * 180 / np.pi, combined_io_corona, label=f"{int(wavelength * 1e3)}nm")

    plt.xlabel('Scattering Angle (degrees)')
    plt.ylabel('Intensity')
    # plt.yscale("log")
    plt.legend()
    plt.show()



if __name__ == "__main__":

    vis_spec =  np.linspace(0.38, 0.7, 3)
    spec = np.array([2.8, 2.816, 2.832])
    # fraunhofer(100, spec, 20, 21)

    # Example usage fraunhofer2:
    particle_radii = [20,20]  # radii in microns
    wavelengths = [2.83, 2.847, 2.864, 2.88 ]
    # wavelengths =[2.88, 2.897, 2.914, 2.93]  # wavelength in microns
    start_deg = 17  # start angle in degrees
    end_deg = 30 # end angle in degrees

    wavelengths = np.array([0.4, 0.45, 0.50, 0.55])
    fraunhofer(30, np.array(wavelengths), -3, 3)
    # fraunhofer2(particle_radii, wavelengths, start_deg, end_deg)

    blue_phase = mie_phase_func("vis", radius=10,start_deg=0, end_deg=25, spacing=0.1, plot=True, peaks=False, norm_type="albedo" )

    # ------------------- ISS filters - Clear, IR3, IR1, RED, BL1 -----------------------------------
    wavelengths = [0.651, 0.928, 0.750, 0.649, 0.455 ]
    colours = ['grey', 'black', 'maroon', 'red', 'blue']
    wavelengths = [ 0.550, 0.6, 0.7, 0.75]
    wavelengths = [ 0.40, 0.43, 0.46, 0.5]

    colours = [ 'red', 'maroon', 'grey', 'black']
    radius = 9.0       # Radius of the particle in micrometers
    # blue_lam, blue_phase = mie_phase_func(wavelengths, radius=3,start_deg=0, end_deg=30, spacing=0.1,colours=colours,plot=True)

    bl1 = np.linspace(0.4, 0.5, 11)
    red = np.linspace(0.55, 0.75, 11)
    ir1 = np.linspace(0.65, 0.85, 11)
    ir3 = np.linspace(0.875, 1.0, 11)

    radius = 5
    start_deg = 0
    end_deg = 30
    spacing = 0.01
    colours = None

    bl_lam, bl_phase = mie_phase_func_band(bl1, radius, start_deg, end_deg, spacing,colours, plot=True)
    red_lam, red_phase = mie_phase_func_band(red, radius, start_deg, end_deg, spacing,colours, plot=False)
    ir1_lam, ir1_phase = mie_phase_func_band(ir1, radius, start_deg, end_deg, spacing,colours, plot=False)
    ir3_lam, ir3_phase = mie_phase_func_band(ir3, radius, start_deg, end_deg, spacing,colours, plot=False)

    plt.plot(bl_lam, bl_phase.T, c='blue', label="BL1")
    plt.plot(red_lam, red_phase.T, c='red', label="RED")
    plt.plot(ir1_lam, ir1_phase.T, c='maroon', label="IR1")
    plt.plot(ir3_lam, ir3_phase.T, c='black', label="IR3")
    plt.legend()
    plt.ylabel("Intensity")
    plt.yscale('log')
    plt.xlabel("Scattering angle [deg]")
    plt.show()