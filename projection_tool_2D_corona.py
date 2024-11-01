import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spiceypy as spice
import csv
from datetime import datetime, timedelta
import math
from plotly.colors import sample_colorscale
from Atlas.search_atlas import main_search
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from Mie_theory.mie_plots import mie_phase_func
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
from projection_defs import *


# ----------------------------- 2d projections -------------------------------------

def plot_phase_projection_single(image_id, instrument_name, particle_radius=1, deg_range=None, overwrite_img=None,
                                 interactive=False, mode="geo", mie_wavelength=None, color_intens=False, save_fig_path=None):
    """
       Plots the phase projection of a given image based on the specified parameters.

       Parameters:
       ----------
       image_id : str
           The identifier for the image to be processed, typically matching the filename (without extension) in the Atlas.

       instrument_name : str
           The name of the instrument used to capture the image (e.g., "CASSINI_ISS_NAC" or "CASSINI_VIMS_IR").

       particle_radius : float, optional
           The radius of the scattering particles in micrometers. Defaults to 1.

       deg_range : list, optional
           The range of degrees for phase angles. If None, defaults to [0, 30].

       overwrite_img : str, optional
           Path to an alternative image to use instead of the default downloaded image. If None, uses the default image.

       interactive : bool, optional
           If True, enables interactive plotting features such as zooming. Defaults to False.

       mode : str, optional
           The mode of operation; "geo" for geometric plotting or "mie" for Mie scattering calculations. Defaults to "geo".

       mie_wavelength : float, optional
           Wavelength of light in nanometers for Mie theory calculations. Required if mode is "mie".

       color_intens : bool, optional
           If True, the color of Mie scattering circles will represent intensity. Defaults to False.

       save_fig_path : str, optional
           If provided, saves the figure to the specified path instead of displaying it.

       Returns:
       -------
       None
           Displays the plot or saves it to the specified path if provided.
       """

    if deg_range is None:
        deg_range = [0, 30]
    if interactive:
        import matplotlib
        matplotlib.use('TkAgg')  # You can replace 'Qt5Agg' with other backends like 'TkAgg' or 'GTK3Agg'


    # search atlas for image and label
    download_dir = "Atlas/downloaded_files"  # Directory to save the downloaded files
    main_search(image_id, download_dir)

    # -------------------------SPICE -------------------------------------------
    print("------------------ SPICE -----------------------")
    METAKR = "./cassMetaK.txt"
    SCLKID = -82
    spice.furnsh(METAKR)

    # extract midtime from atlas label
    utc_mid_time = extract_value_from_label(f"Atlas/downloaded_files/{image_id}.lbl", "IMAGE_MID_TIME")
    print(utc_mid_time)
    print("Mid time UTC = ", utc_mid_time)
    et = spice.str2et(utc_mid_time)
    print("ET seconds past J2: {} ".format(et))

    # get fov
    room = 4  # the maximum number of 3-dimensional vectors that can be returned in `bounds'.
    [shape, insfrm, bsight, n, bounds] = spice.getfvn(instrument_name, room)
    bounds = np.vstack((bounds, [0, 0, 0]))  # add the origin of the fov (tip of the pyramid)
    fov_angle = spice.convrt(spice.vsep(bounds[1], bounds[0]), 'RADIANS', 'DEGREES')

    # get solar pos as observed from cassini in the instrument frame
    sun_pos, ltime_iss = spice.spkpos('SUN', et, instrument_name, 'LT+S', 'CASSINI')

    # get colatitude and longitude in spherical coordinates
    r, colat, slon = spice.recsph(sun_pos)
    colat_deg = np.rad2deg(colat)  # angle between point and pos z-axis in radians

    # set latitudes negative higher than 90 degrees ( not necceassry want gnomonic projection
    if colat_deg > 90:
        # colat_deg = (colat_deg - 90) * -1
        colat_deg = 90
    slon_deg = np.rad2deg(slon)  # longitude in radians
    # right_ascension_deg = slon_deg if slon_deg >= 0 else slon_deg + 360  # Ensure RA is in [0, 360]

    # y-axis heigh is equal to colat angle
    point = [colat_deg, 0]

    # RA represents clockwise rotation around z-axis
    # function does counter clockwise rot but assume right hand cord system (here we have left so it cancels out)
    point_rotate = rotate([0, 0], point, np.deg2rad(slon_deg))

    scat_angle = spice.convrt(spice.vsep(bsight, sun_pos), 'RADIANS', 'DEGREES')
    print("Scattering angle=", scat_angle)
    sun_cord_og = point_rotate
    sun_cord = np.rad2deg(stereographic_projection(np.deg2rad(point_rotate[1]), np.deg2rad(point_rotate[0])))

    # angle between x-axis and sun-vector
    sun_angle = angle_between_vectors([-1, 0], sun_cord_og)
    print("sun angle with pos x-axis", sun_angle)

    # Ecliptic
    # Define the ecliptic frame
    ecliptic_frame = 'ECLIPJ2000'

    # Vernal Equinox (RA = 0°, Dec = 0° in ECLIPJ2000)
    vernal_equinox_ra_dec = [0.0, 0.0]  # (RA, Dec) in degrees

    # Autumnal Equinox (RA = 180°, Dec = 0° in ECLIPJ2000)
    autumnal_equinox_ra_dec = [180.0, 0.0]  # (RA, Dec) in degrees

    # Convert the RA/Dec to unit vectors in the ecliptic frame
    def ra_dec_to_vector(ra_deg, dec_deg):
        ra_rad = np.deg2rad(ra_deg)
        dec_rad = np.deg2rad(dec_deg)
        return [
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad)
        ]

    vernal_vector = ra_dec_to_vector(*vernal_equinox_ra_dec)
    autumnal_vector = ra_dec_to_vector(*autumnal_equinox_ra_dec)

    # Get transformation matrix from ECLIPJ2000 to the instrument frame
    instrument_to_ecliptic_matrix = spice.pxform(ecliptic_frame, instrument_name, et)

    # Transform the vectors to the instrument frame
    vernal_instrument = spice.mxv(instrument_to_ecliptic_matrix, vernal_vector)
    autumnal_instrument = spice.mxv(instrument_to_ecliptic_matrix, autumnal_vector)

    # Convert the vectors to spherical coordinates for plotting (colatitude, longitude)
    r_vernal, colat_vernal, lon_vernal = spice.recsph(vernal_instrument)
    r_autumnal, colat_autumnal, lon_autumnal = spice.recsph(autumnal_instrument)

    vernal_point = [np.rad2deg(colat_vernal), 0]  # Define a point for plotting, with zero azimuth
    vernal_point_rotated = rotate([0, 0], vernal_point, lon_vernal)

    aut_point = [np.rad2deg(colat_autumnal), 0]  # Define a point for plotting, with zero azimuth
    aut_point_rotated = rotate([0, 0], aut_point, lon_autumnal)

    # Clean up the kernels
    spice.kclear()

    # ------------------------------ Mie theory --------------------------------
    # create scattering circles
    scat_radii = np.arange(0, 51, 5)
    scat_radii_rad = np.deg2rad(scat_radii)
    # convert to steroegraphic projection
    # scat_radii = [stereographic_projection(0, angle)[0] for angle in scat_radii_rad]
    # scat_radii = np.rad2deg(scat_radii)

    if mode == "geo":
        # Add extra circles with color around fov
        range_boundary = fov_angle / 1.5
        center_angle = scat_angle
        # define a range of circle radii close to the fov
        scat_bounds = np.linspace(center_angle - range_boundary, center_angle + range_boundary, 20)
        scat_bounds_rad = np.deg2rad(scat_bounds)

    if mode == "mie":
        mie_theta, mie_intens, mie_color = mie_phase_func(wavelengths=mie_wavelength, radius=particle_radius,
                                                          start_deg=deg_range[0],
                                                          end_deg=deg_range[1], spacing=fov_angle / 40, plot=True,
                                                          norm_type="albedo", color_intensity=color_intens,
                                                          material_data_link="opt_cons/warren_2008_ASCIItable.txt")

        # # Create a dictionary to map each mie_step to its corresponding color
        mie_color_dict = {theta: color for theta, color in zip(mie_theta, mie_color)}

    # --------------Create figure and axes -----------------------------
    fig, ax = plt.subplots()

    #  Add background image
    img_path = f"Atlas/downloaded_files/{image_id}.jpeg"
    if overwrite_img:
        img_path = overwrite_img
    img = plt.imread(img_path)
    image_extent = [fov_angle/2, -fov_angle/2, -fov_angle/2, fov_angle/2]
    img = ax.imshow(img, extent=image_extent, aspect='auto', alpha=1, zorder=-1)
    img.set_cmap("grey")

    if mode =="geo":
        # Define colorscale
        norm = Normalize(vmin=min(scat_bounds), vmax=max(scat_bounds))
        colors = cm.viridis(norm(scat_bounds))

        # Add extra circles with colors
        for i, bound in enumerate(scat_bounds):
            circle = Circle((sun_cord_og[0], sun_cord_og[1]), bound, edgecolor=colors[i], facecolor='none', linestyle='-',
                            linewidth=1)
            ax.add_patch(circle)

        # Adding a colorbar for the extra circles
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.1)
        cbar.set_label('scattering angle (deg)')

    if mode == "mie":
        # Add extra circles with colors
        # for i, mie_step in enumerate(mie_theta):
        #     circle = Circle((sun_cord_og[0], sun_cord_og[1]), mie_step, edgecolor=mie_color[i], facecolor='none',
        #                     linestyle='-',
        #                     linewidth=3)
        #     ax.add_patch(circle)

        # Define the range of interest for high resolution
        high_res_range = (scat_angle-1.5*fov_angle, scat_angle+1.5*fov_angle)  # Example: Keep high resolution between 20 and 100 degrees
        low_res_interval = 5  # Draw one out of every `low_res_interval` circles outside the high-resolution range

        # Separate the mie_theta values into high-res and low-res
        high_res_steps = [theta for theta in mie_theta if high_res_range[0] <= theta <= high_res_range[1]]
        low_res_steps = [theta for theta in mie_theta if theta < high_res_range[0] or theta > high_res_range[1]]

        # Add circles for high-resolution steps
        for theta in high_res_steps:
            color = mie_color_dict[theta]
            circle = Circle((sun_cord_og[0], sun_cord_og[1]), theta, edgecolor=color, facecolor='none',
                            linestyle='-', linewidth=2)
            ax.add_patch(circle)

        # Add circles for downsampled low-resolution steps
        for i, theta in enumerate(low_res_steps):
            if i % low_res_interval == 0:
                color = mie_color_dict[theta]
                circle = Circle((sun_cord_og[0], sun_cord_og[1]), theta, edgecolor=color, facecolor='none',
                                linestyle='-', linewidth=10)
                ax.add_patch(circle)


    # Add main circles
    for radius in scat_radii:
        circle = Circle((sun_cord_og[0], sun_cord_og[1]), radius, edgecolor='black', facecolor='none', linestyle='-',
                        linewidth=1)
        ax.add_patch(circle)
        # Calculate the top center of the circle
        text_x = sun_cord_og[0]
        text_y = sun_cord_og[1] + radius
        # Add text annotation for the radius
        ax.text(text_x, text_y, f'{radius}°', color='black', fontsize=8, ha='center', va='bottom')

    #add fov
    rect = Rectangle((image_extent[1], image_extent[2]), fov_angle, fov_angle, edgecolor='black', facecolor='none',
                     linestyle='-', linewidth=2)
    ax.add_patch(rect)

    # Add a grid to the plot
    ax.grid(True, which='both', color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot "Sun spice" and "Boresight" points
    ax.scatter([sun_cord_og[0]], [sun_cord_og[1]], label="Sun", color="yellow")
    ax.scatter([0], [0], label="Boresight", color="red")
    ax.plot([0, sun_cord_og[0]], [0, sun_cord_og[1]], color='yellow', linestyle='--',
            label="Sun-bsight vector", zorder=1)


    if instrument_name == "CASSINI_ISS_NAC" or "CASSINI_ISS_WAC":
        ax.set_xlim(fov_angle / 2, -fov_angle / 2)
        ax.set_ylim(-fov_angle / 2, fov_angle / 2)
        # ax.set_xlim(-fov_angle / 2, fov_angle / 2)
        # ax.set_ylim(-fov_angle / 2, fov_angle / 2)
    if instrument_name == "CASSINI_VIMS_IR":
        ax.set_xlim(-fov_angle / 2, fov_angle / 2)
        ax.set_ylim(fov_angle / 2, -fov_angle / 2)
    ax.set_aspect('equal')
    ax.set_xlabel('x fov (deg)')
    ax.set_ylabel('y fov (deg)')

    # Title and labels
    ax.set_title(f'{image_id}')
    # ax.xlabel('x fov (deg)')
    # plt.ylabel('y fov (deg)')


    # Enable interactive zooming
    def zoom_factory(ax, base_scale=2.):
        def zoom_fun(event):
            # Get the current axis limits
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata  # get event x location
            ydata = event.ydata  # get event y location

            if event.button == 'up':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])
            plt.draw()  # force re-draw

        fig.canvas.mpl_connect('scroll_event', zoom_fun)
        return zoom_fun

    # Apply zooming function
    zoom_factory(ax)
    # Show the plot
    if save_fig_path:
        plt.savefig(save_fig_path)

    ax.legend(loc="lower left", facecolor=None, edgecolor=None)
    # plt.legend()

    plt.show()

    return


def plot_phase_projection_multi(flyby_name, utc_start, utc_end, time_steps, instrument_name, particle_radius=1,
                                deg_range=None, interactive=False,  mie_wavelength=None, color_intens=False):
    """
        Plot the phase projection of scattering angles for a specified instrument during a spacecraft flyby.

        This function uses SPICE kernels to compute the position of the Sun and the field of view (FOV) of the specified
        instrument during a given time range. It creates a scatter plot that visualizes the scattering angles based on
        Mie scattering theory and allows for interactive time navigation through a slider.

        Parameters:
        ----------
        flyby_name : str
            The name of the flyby event to analyze.
        utc_start : str
            The start time of the observation period in UTC format (e.g., "YYYY-MM-DD HH:MM:SS").
        utc_end : str
            The end time of the observation period in UTC format.
        time_steps : int
            The number of time steps to consider for the analysis.
        instrument_name : str
            The name of the instrument used for the observations (e.g., "CASSINI_ISS_WAC").
        particle_radius : float, optional
            The radius of the scattering particles in micrometers (default is 1).
        deg_range : list of float, optional
            The range of scattering angles in degrees (default is [0, 30]).
        interactive : bool, optional
            If True, enables an interactive plot that allows navigation through time (default is False).
        mie_wavelength : float, optional
            The wavelength of light used for Mie scattering calculations in micrometers (default is None).
        color_intens : bool, optional
            If True, colors the scattering circles based on intensity, otherwise uses a uniform color (default is False).

        Returns:
        -------
        None
            The function displays a plot but does not return any value.

        Notes:
        -----
        The function requires the `matplotlib` and `spice` libraries for plotting and SPICE kernel handling, respectively.
        Ensure that the SPICE kernels are correctly set up before calling this function.
        """

    if deg_range is None:
        deg_range = [0, 30]
    if interactive:
        import matplotlib
        matplotlib.use('TkAgg')


    # -------------------------SPICE -------------------------------------------
    print("------------------ SPICE -----------------------")
    METAKR = "./cassMetaK.txt"
    SCLKID = -82
    spice.furnsh(METAKR)

    et_start = spice.utc2et(utc_start)
    et_end = spice.utc2et(utc_end)
    print("ET start seconds past J2: {} ".format(et_start))
    print("ET end seconds past J2: {} ".format(et_end))

    # time steps
    times_filtered = get_remaining_ets(flyby_name, time_steps, start_et=et_start, end_et=et_end)
    times_filtered_utc = spice.et2utc(times_filtered, "C", 0)

    # get fov
    room = 4  # the maximum number of 3-dimensional vectors that can be returned in `bounds'.
    [shape, insfrm, bsight, n, bounds] = spice.getfvn(instrument_name, room)
    bounds = np.vstack((bounds, [0, 0, 0]))  # add the origin of the fov (tip of the pyramid)
    fov_angle = spice.convrt(spice.vsep(bounds[1], bounds[0]), 'RADIANS', 'DEGREES')

    # get solar pos as observed from cassini in the instrument frame
    sun_pos, ltime_iss = spice.spkpos('SUN', np.array(times_filtered), instrument_name, 'LT+S', 'CASSINI')

    sun_cord_list = []
    scat_angle_list = []
    for i, pos in enumerate(sun_pos):
        # get colatitude and longitude in spherical coordinates
        r, colat, slon = spice.recsph(pos)
        colat_deg = np.rad2deg(colat)  # angle between point and pos z-axis in radians

        # set latitudes negative higher than 90 degrees ( not necceassry want gnomonic projection
        if colat_deg > 90:
            # colat_deg = (colat_deg - 90) * -1
            colat_deg = 90
        slon_deg = np.rad2deg(slon)  # longitude in radians

        # y-axis heigh is equal to colat angle
        point = [colat_deg, 0]
        # RA represents clockwise rotation around z-axis
        # function does counter clockwise rot but assume right hand cord system (here we have left so it cancels out)
        point_rotate = rotate([0, 0], point, np.deg2rad(slon_deg))

        sun_cord_og = point_rotate
        sun_cord_list.append(sun_cord_og)

        scat_angle = spice.convrt(spice.vsep(bsight, pos), 'RADIANS', 'DEGREES')
        scat_angle_list.append(scat_angle)

    # ------------------------------ Mie theory --------------------------------
    # create scattering circles
    scat_radii = np.arange(0, 31, 5)
    scat_radii_rad = np.deg2rad(scat_radii)
    if instrument_name == "CASSINI_ISS_WAC":
        fov_steps = 20
        linew = 20
    elif instrument_name == "CASSINI_ISS_NAC":
        fov_steps = 15
        linew = 24
    else:
        fov_steps = 15
        linew = 24

    mie_theta, mie_intens, mie_color = mie_phase_func(wavelengths=mie_wavelength, radius=particle_radius,
                                                      start_deg=deg_range[0],
                                                      end_deg=deg_range[1], spacing=fov_angle / fov_steps, plot=True,
                                                      norm_type="albedo", color_intensity=color_intens,
                                                      material_data_link="opt_cons/warren_2008_ASCIItable.txt")

    # # Create a dictionary to map each mie_step to its corresponding color
    mie_color_dict = {theta: color for theta, color in zip(mie_theta, mie_color)}

    # --------------Create figure and axes -----------------------------

    # Function to update the plot based on slider value
    def update(val):
        ax.clear()
        index = int(slider.val)

        scat_angle = scat_angle_list[index]
        sun_cord = sun_cord_list[index]

        # Define the range of interest for high resolution
        high_res_range = (scat_angle - fov_angle, scat_angle + fov_angle)
        # Separate the mie_theta values into high-res and low-res
        high_res_steps = [theta for theta in mie_theta if high_res_range[0] <= theta <= high_res_range[1]]

        # Add circles for high-resolution steps
        for theta in high_res_steps:
            color = mie_color_dict[theta]
            circle = Circle((sun_cord[0], sun_cord[1]), theta, edgecolor=color, facecolor='none',
                            linestyle='-', linewidth=linew)
            ax.add_patch(circle)
            # circle = Circle((sun_cord[0], sun_cord[1]), theta, edgecolor="grey", facecolor='none',
            #                 linestyle='-', linewidth=1)
            # ax.add_patch(circle)

        # Add grid
        ax.grid(True, which='both', color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

        # Plot "Sun spice" and "Boresight" points
        ax.scatter([0], [0], label="Boresight", color="red")
        ax.plot([0, sun_cord[0]], [0, sun_cord[1]], color='yellow', linestyle='--',
                label="Sun-bsight vector", zorder=1)

        # Set plot limits
        if instrument_name == "CASSINI_ISS_NAC" or "CASSINI_ISS_WAC":
            ax.set_xlim(fov_angle / 2, -fov_angle / 2)
            ax.set_ylim(-fov_angle / 2, fov_angle / 2)
            # ax.set_xlim(-fov_angle / 2, fov_angle / 2)
            # ax.set_ylim(-fov_angle / 2, fov_angle / 2)
        if instrument_name == "CASSINI_VIMS_IR":
            ax.set_xlim(-fov_angle / 2, fov_angle / 2)
            ax.set_ylim(fov_angle / 2, -fov_angle / 2)
        ax.set_aspect('equal')
        ax.set_xlabel('x fov (deg)')
        ax.set_ylabel('y fov (deg)')

        # Update title
        ax.set_title(f"Center scattering angle: {np.round(scat_angle_list[index],2)}")
        # Update the text box with the current timestamp
        time_text.set_text(times_filtered_utc[index])

        # Show the legend
        ax.legend()

        # Redraw the figure canvas
        fig.canvas.draw_idle()

    # Initial plot setup
    fig, ax = plt.subplots(figsize=(10, 6))

    # Slider for time navigation
    ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Time', 0, len(times_filtered) - 1, valinit=0, valstep=1)
    # Text box to display the current timestamp
    time_text = fig.text(0.9, 0.02, times_filtered_utc[0], fontsize=6, ha='left')

    # Attach update function to slider
    slider.on_changed(update)

    # Display the initial plot
    update(0)

    plt.show()
    # Clean up the kernels
    spice.kclear()
    return


# ------------------------------------------------------------------------------------

# retrieve a single observation for image atlas
plot_phase_projection_single("N1711553974", "CASSINI_ISS_NAC", overwrite_img=None, interactive=True, mode="geo",
                             particle_radius=3, deg_range=[0,25], mie_wavelength="vis", color_intens=False)

# select time range of obervations
start = "2012/03/27 14:38"
end = "2012/03/27 14:45"
# or
# flyby_name = "E18"
# utc_range = get_time_flyby(flyby_name, "flybys.txt", 10)
# start = utc_range[0]
# end = utc_range[1]
print("start UTC = ", start)
print("end UTC = ", end)
plot_phase_projection_multi("E17", utc_start=start, utc_end=end, time_steps=10, instrument_name="CASSINI_VIMS_IR",
                            particle_radius=10 ,interactive=True, deg_range=[15,30], mie_wavelength=[0.65],
                            color_intens=True)


