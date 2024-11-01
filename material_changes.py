import matplotlib
matplotlib.use('TkAgg')  # You can replace 'Qt5Agg' with other backends like 'TkAgg' or 'GTK3Agg'
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc
import scipy.special as scp
from scipy import signal
import pandas as pd


def halo_angle(n_halo):
    refracting_angle = 60 * np.pi / 180
    d_min = 2 * np.arcsin(n_halo * np.sin(refracting_angle / 2)) - refracting_angle
    d_min_deg = list(d_min * 180 / np.pi)
    return d_min_deg


def halo_angle2n(d_min_deg):
    refracting_angle = 60 * np.pi / 180
    d_min = np.array(d_min_deg) * np.pi / 180
    n_halo = np.sin((d_min+refracting_angle)/2)/np.sin(refracting_angle/2)
    return n_halo


def rainbow_angle(n_rainbow):
    k = 1
    rainbow_incidence = np.arccos(np.sqrt((n_rainbow ** 2 - 1) / (k * (2 + k))))
    rainbow_refract = np.arcsin(np.sin(rainbow_incidence) / n_rainbow)
    rainbow_scat = (2 * rainbow_incidence - 2 * (1 + k) * rainbow_refract) * 180 / np.pi
    rainbow_scat = list(rainbow_scat)
    # print("rainbow scattering angles [deg]:", rainbow_scat)
    return rainbow_scat

# range of particle sizes from 1-10 -> uv  to ir
def corona_angle(particle_radius, wavelength):
    """

    Parameters
    ----------
    particle_radius in micron
    wavelength in nm

    Returns
    -------

    """
    cor_angle_range = 40
    corona_scat = np.linspace(0, cor_angle_range, cor_angle_range * 1000) * np.pi / 180  # scattering angle range
    wavelength = wavelength * 1e-9
    r = particle_radius*1e-6  # particle radius
    x = list(2 * np.pi * r / wavelength)  # size parameter
    wavelength = list(wavelength)
    plt.figure()
    for j in range(len(x)):
        bes_argument = x[j] * np.sin(corona_scat)
        j1 = scp.jv(1, bes_argument)  # Using scipy.special.jv for Bessel function of the first kind
        io_corona = (x[j] * (1 + np.cos(corona_scat)) / 2 * j1 / bes_argument) ** 2
        # Plot the Bessel function
        plt.plot(corona_scat * 180 / np.pi, io_corona, label=f"{int(wavelength[j] * 1e9)}nm")
        # Find peaks in the data
        peaks, _ = sc.signal.find_peaks(io_corona)
        # Select the scattering angles of the first 3 peaks
        scattering_angles = corona_scat[peaks[:3]]
        print(f"corona scattering angles of the first 3 maxima for {wavelength[j]}:",
              scattering_angles * 180 / np.pi, "degrees")
    return
# ---------------------------- wavelenghts and materials --------------------------

# combine ice data:
# Combine UVIS and NIR data into one DataFrame
# ice_data_comb = pd.concat([ice_uvis_data[['lambda', 'n']], ice_nir_data[['lambda', 'n']]], ignore_index=True)
# # Sort the DataFrame by 'lambda' column
# ice_data_comb = ice_data_comb.sort_values(by='lambda').reset_index(drop=True)

# Warren 2008: n-ice UV to IR crystalline at room temp T=266K
ice_data_warren = pd.read_csv('opt_cons/warren_2008_ASCIItable.txt', sep='\s+', header=None)
ice_data_warren.columns =['lambda', 'n', 'k']
ice_data_warren['lambda']= ice_data_warren['lambda']*1000     # convert to nm

# Segelstein: n liquid water  10nm - >10 micron
water_data = pd.read_csv('opt_cons/Segelstein.txt', sep="\t", header=2)
water_data.columns = ["lambda", "n", "k"]
water_data['lambda'] = water_data['lambda']*1000        # convert to nm

# Mastrapa SWIR 2009 covers 1-5 micron
# Define the file path
file_path = "opt_cons/Mastrapa_nk.txt"
skip_rows = 30
column_names = ["Form", "Temp", "lambda", "n", "k"]
mastrapa_swir = pd.read_csv(file_path, skiprows=skip_rows, sep='\s+', names=column_names)
mastrapa_swir['lambda'] = mastrapa_swir['lambda']*1000
temps_am =[15, 25, 40, 50, 60, 80, 100, 120]
temps_crys = np.arange(20, 151, 10)
temps_am = [15, 120]
temps_crys = [20, 150]

# He et al 2022
ice_vis_he130 = pd.read_csv('opt_cons/He_135K.txt', delim_whitespace=True, names=['wavenumber', 'n'])
ice_vis_he160 = pd.read_csv('opt_cons/he_etal_160k_self_extract.csv', header=None, names=['lambda', 'n'])
# Convert wavenumber to wavelength in nanometers
ice_vis_he130['lambda'] = 10**7 / ice_vis_he130['wavenumber']

# Rocha 2024
ice_rocha160 = pd.read_csv('opt_cons/rocha160k.csv', header=None, names=['lambda', 'n'])
ice_rocha160['lambda']= ice_rocha160['lambda']*1000 # in nm
# ice refractive indices
plt.figure()

for i in range(len(temps_crys)):
    mastrapa_crys_i = mastrapa_swir[(mastrapa_swir["Form"] == "Crystalline") & (mastrapa_swir["Temp"] == temps_crys[i])]
    plt.plot(mastrapa_crys_i["lambda"]/1000, mastrapa_crys_i["n"], label="Crystalline {}K (Mastrapa 2009)".format(temps_crys[i]) ,linestyle='-')

# Kofman 2019, n ice in uvis range: 200-757 nm
ice_uvis_data = pd.read_csv('opt_cons/kofman_uvis_file.txt', sep=" ", header=21)
ice_uvis_data = ice_uvis_data.drop(ice_uvis_data.columns[-1], axis=1)
names = ['lambda', 'Rl', 'e_Rl', 'nl-10K', 'nl-30K', 'nl-50K', 'nl-70K', 'nl-90K', 'nl-110K', 'nl-130K', 'nl-150K']
ice_uvis_data.columns = names
# plt.figure()
plt.plot(ice_data_warren['lambda']/1000, ice_data_warren["n"], label="Crystalline 266K (Warren 2008)")
plt.plot(ice_uvis_data['lambda']/1000, ice_uvis_data['nl-150K'], label="Crystalline 150 K (Kofman 2019)")
plt.plot(ice_uvis_data['lambda']/1000, ice_uvis_data['nl-130K'], label="Amorphous 130K (Kofman 2019)")
plt.plot(ice_vis_he160['lambda']/1000, ice_vis_he160['n'], label="Crystalline 160K (He 2022)")
plt.plot(ice_rocha160['lambda']/1000, ice_rocha160['n'], label="Crystalline 160K (Rocha 2024)")

# plt.title("n ice visible spectrum")
plt.xlabel(r"wavelength $\mu$m")
plt.xlim(0.2,5)
plt.ylabel("n")
plt.legend()
plt.show()

# halo
# cannot load in data from Jiao He for crystalline ice 160k in the vis
# from observations +-1 degree up in refractive index 22->23
lambdas = ice_uvis_data['lambda']
vis_halo_angles_150k = halo_angle(ice_uvis_data['nl-150K'])
vis_halo_angles_130k =  halo_angle(ice_uvis_data['nl-130K'])

mastrapa_crys_20k = mastrapa_swir[(mastrapa_swir["Form"] == "Crystalline") & (mastrapa_swir["Temp"] == 20)]
mastrapa_crys_150k = mastrapa_swir[(mastrapa_swir["Form"] == "Crystalline") & (mastrapa_swir["Temp"] == 150)]
ir_halo_angles_20k = halo_angle(mastrapa_crys_20k["n"])
ir_halo_angles_150k = halo_angle(mastrapa_crys_150k["n"])

plt.plot(mastrapa_crys_20k['lambda'], ir_halo_angles_20k, label="Crys 20K (Mastrapa 2009)")
plt.plot(mastrapa_crys_150k['lambda'], ir_halo_angles_150k, label="Crys 150K (Mastrapa 2009)")
plt.plot(lambdas, vis_halo_angles_150k, label="Crys 150K (Kofman 2019)")
plt.plot(ice_vis_he160['lambda'], halo_angle(ice_vis_he160['n']), label="Crys 160K (He 2022)")
plt.plot(ice_rocha160['lambda'], halo_angle(ice_rocha160['n']), label="Crys 160K (Rocha 2024)")
plt.plot(ice_data_warren['lambda'], halo_angle(ice_data_warren['n']), label="Crys 266K (Warren 2008)", color='black')
plt.plot(lambdas, vis_halo_angles_130k, label="Amorph 130K (Kofman 2019)")

# plt.title("Halo scattering angle")
plt.ylabel(r"Halo scattering angle [$^\circ$]")
plt.xlabel("Wavelength [nm]")
plt.xlim(200, 5000)
# plt.ylim(10, 35)
plt.legend()
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First subplot: full wavelength range
ax1.plot(mastrapa_crys_20k['lambda'], ir_halo_angles_20k, label="Crys 20K (Mastrapa 2009)")
ax1.plot(mastrapa_crys_150k['lambda'], ir_halo_angles_150k, label="Crys 150K (Mastrapa 2009)")
ax1.plot(lambdas, vis_halo_angles_150k, label="Crys 150K (Kofman 2019)")
ax1.plot(ice_vis_he160['lambda'], halo_angle(ice_vis_he160['n']), label="Crys 160K (He 2022)")
ax1.plot(ice_rocha160['lambda'], halo_angle(ice_rocha160['n']), label="Crys 160K (Rocha 2024)")
ax1.plot(ice_data_warren['lambda'], halo_angle(ice_data_warren['n']), label="Crys 266K (Warren 2008)", color='black')
ax1.plot(lambdas, vis_halo_angles_130k, label="Amorph 130K (Kofman 2019)")

ax1.set_ylabel(r"Halo scattering angle [$^\circ$]")
ax1.set_xlabel("Wavelength [nm]")
ax1.set_xlim(200, 5000)
ax1.legend()

# Second subplot: zoomed in
ax2.plot(mastrapa_crys_20k['lambda'], ir_halo_angles_20k, label="Crys 20K (Mastrapa 2009)")
ax2.plot(mastrapa_crys_150k['lambda'], ir_halo_angles_150k, label="Crys 150K (Mastrapa 2009)")
ax2.plot(lambdas, vis_halo_angles_150k, label="Crys 150K (Kofman 2019)")
ax2.plot(ice_vis_he160['lambda'], halo_angle(ice_vis_he160['n']), label="Crys 160K (He 2022)")
ax2.plot(ice_rocha160['lambda'], halo_angle(ice_rocha160['n']), label="Crys 160K (Rocha 2024)")
ax2.plot(ice_data_warren['lambda'], halo_angle(ice_data_warren['n']), label="Crys 266K (Warren 2008)", color='black')
ax2.plot(lambdas, vis_halo_angles_130k, label="Amorph 130K (Kofman 2019)")

ax2.set_xlabel("Wavelength [nm]")
ax2.set_xlim(200, 1800)  # Zoomed in view
ax2.set_ylim(15, 25)  # Zoomed in view

ax2.legend()

plt.tight_layout()
plt.show()

# rainbow

#ocean optics book
def ocean_optics_n_water(temperature, wavelength, salinity):
    """
    https://oceanopticsbook.info/view/optical-constituents-of-the-ocean/water
    Parameters
    ----------
    temperature in celsius min 0 max 30
    wavelength in nm, min 400 max 700, for S=0 200-1100 nm
    salinity in PSU (practical salinity), sea water = 35

    Returns
    -------

    """
    # avalaible for 400-700nm, T = 0-30 Celsius,
    T =  temperature
    S = salinity
    lam = wavelength

    n0 = 1.31405
    n1 = 1.779*1e-4
    n2 = -1.05*1e-6
    n3 = 1.6*1e-8
    n4 = -2.02*1e-6
    n5 = 15.868
    n6 = 0.01155
    n7 = -0.00423
    n8 = -4382
    n9 = 1.1455*1e6

    n = n0 + (n1 +n2*T +n3*T**2)*S + n4*T**2 + (n5 +n6*S +n7*T)/lam + n8/lam**2 + n9/lam**3
    return n


def index_of_refraction_spectrum(temperature, wavelength_range, water_type):
    """
    Compute the index of refraction of saltwater or freshwater for a range of wavelengths.
    #Christopher Parrish (2020)
    Parameters:
        temperature (float): Temperature in degrees Celsius (valid range: 0-30).
        wavelength_range (tuple): Tuple containing start and end wavelengths in nanometers
                                  (valid range: 400-700).
        water_type (str): Type of water, either 'seawater' or 'freshwater'.

    Returns:
        dict: Dictionary containing wavelengths as keys and corresponding indices of refraction as values.
    """
    # Coefficients for seawater and freshwater
    coefficients = {
        'seawater': {
            'a': -1.50156e-6,
            'b': 1.07085e-7,
            'c': -4.27594e-5,
            'd': -1.60476e-4,
            'e': 1.39807
        },
        'freshwater': {
            'a': -1.97812e-6,
            'b': 1.03223e-7,
            'c': -8.58125e-6,
            'd': -1.54834e-4,
            'e': 1.38919
        }
    }

    # Validate input temperature
    if not (0 <= temperature <= 30):
        raise ValueError("Temperature must be in the range 0-30 degrees Celsius.")

    # Validate input wavelength range
    start_wavelength, end_wavelength = wavelength_range
    if not (400 <= start_wavelength <= 700) or not (400 <= end_wavelength <= 700):
        raise ValueError("Wavelength range must be in the range 400-700 nanometers.")

    # Generate wavelengths and compute corresponding index of refraction
    wavelengths = range(start_wavelength, end_wavelength + 1)
    indices_of_refraction = []

    for wavelength in wavelengths:
        n = (coefficients[water_type]['a'] * temperature ** 2 +
             coefficients[water_type]['b'] * wavelength ** 2 +
             coefficients[water_type]['c'] * temperature +
             coefficients[water_type]['d'] * wavelength +
             coefficients[water_type]['e'])
        indices_of_refraction.append(round(n, 4))

    return indices_of_refraction

# Example usage:
# temperature = 25  # degrees Celsius
# wavelength_range = (400, 700, 1)  # nanometers
# wavelength_range_nosalt =  (200, 1100, 1)
# indices_of_refraction_salt = index_of_refraction_spectrum(temperature, wavelength_range, 'seawater')
# indices_of_refraction = index_of_refraction_spectrum(temperature, wavelength_range, 'freshwater')
# indices_of_refraction_0 = index_of_refraction_spectrum(0, wavelength_range, 'freshwater')
# indices_of_refraction_0s = index_of_refraction_spectrum(0, wavelength_range, 'seawater')
# plt.plot(np.arange(400, 701, 1), rainbow_angle(np.array(indices_of_refraction)), label="fresh water 25C")
# plt.plot(np.arange(400, 701, 1), rainbow_angle(np.array(indices_of_refraction_salt)),  label="seawater 25C")
# plt.plot(np.arange(400, 701, 1), rainbow_angle(np.array(indices_of_refraction_0)),  label="fresh water 0C")
# plt.plot(np.arange(400, 701, 1), rainbow_angle(np.array(indices_of_refraction_0s)),  label="sea water 0C")

temperature = 25  # degrees Celsius
wavelength_range = np.arange(400, 700, 1)  # nanometers
wavelength_range_nosalt =  np.arange(200, 1100, 1)
indices_of_refraction0 = ocean_optics_n_water(0, wavelength_range_nosalt, 0) # S=0 , temp=0
indices_of_refraction30 = ocean_optics_n_water(30, wavelength_range_nosalt, 0) # S=0 , temp=0
indices_of_refraction0s = ocean_optics_n_water(0, wavelength_range, 35) # S=0 , temp=0
indices_of_refraction30s = ocean_optics_n_water(30, wavelength_range, 35) # S=0 , temp=0

n_rainbow = np.array(water_data['n'])
plt.plot(wavelength_range_nosalt, rainbow_angle(np.array(indices_of_refraction0)), label="fresh water 273K (Quan and Fry 1995)")
plt.plot(water_data['lambda'], rainbow_angle(n_rainbow), label="fresh water 298K (Segesltein 1971)", color='black')
plt.plot(wavelength_range_nosalt, rainbow_angle(np.array(indices_of_refraction30)),  label="fresh water 303K (Quan and Fry 1995)")
plt.plot(wavelength_range, rainbow_angle(np.array(indices_of_refraction0s)),  label="sea water 273K (Quan and Fry 1995)")
plt.plot(wavelength_range, rainbow_angle(np.array(indices_of_refraction30s)),  label="sea water 303K (Quan and Fry 1995)")
plt.ylim( -45, -37)
plt.xlim(300, 1100)
plt.ylabel(r"Rainbow backscatter angle [$^\circ$]")
plt.xlabel("Wavelength [nm]")
plt.legend()


plt.show()

