"""

ariel@iag
04/02/2020

"Hidden functions" to use in LAPIS classes

"""

import bagpipes as pipes
import numpy as np
import matplotlib.pyplot as plt


def make_bagpipes_model(age, tau, mass, metallicity, Av, z):
    """

    Plots a BAGPIPES model for S-PLUS filters

    """

    exp = {}
    exp["age"] = age
    exp["tau"] = tau
    exp["massformed"] = mass
    exp["metallicity"] = metallicity

    dust = {}
    dust["type"] = "Calzetti"
    dust["Av"] = Av
    dust["eta"] = 2.

    nebular = {}
    nebular["logU"] = -2.

    model_components = {}
    model_components["redshift"] = z
    model_components["exponential"] = exp
    model_components["dust"] = dust
    model_components["t_bc"] = 0.01
    model_components["nebular"] = nebular

    filter_list_splus = ['./filters/F378.dat',
                         './filters/F395.dat',
                         './filters/F410.dat',
                         './filters/F430.dat',
                         './filters/F515.dat',
                         './filters/F660.dat',
                         './filters/F861.dat',
                         './filters/u_SPLUS.dat',
                         './filters/g_SPLUS.dat',
                         './filters/r_SPLUS.dat',
                         './filters/i_SPLUS.dat',
                         './filters/z_SPLUS.dat']

    model = pipes.model_galaxy(model_components, filt_list=filter_list_splus)

    return model


def read_splus_photometry(galaxy_table, galaxy_index):
    """

    Reads S-PLUS photometry

    """

    magnitudes = np.array([galaxy_table['F378_auto'][galaxy_index],
                           galaxy_table['F395_auto'][galaxy_index],
                           galaxy_table['F410_auto'][galaxy_index],
                           galaxy_table['F430_auto'][galaxy_index],
                           galaxy_table['F515_auto'][galaxy_index],
                           galaxy_table['F660_auto'][galaxy_index],
                           galaxy_table['F861_auto'][galaxy_index],
                           galaxy_table['uJAVA_auto'][galaxy_index],
                           galaxy_table['g_auto'][galaxy_index],
                           galaxy_table['r_auto'][galaxy_index],
                           galaxy_table['i_auto'][galaxy_index],
                           galaxy_table['z_auto'][galaxy_index]])

    magnitude_errors = np.array([galaxy_table['eF378_auto'][galaxy_index],
                                 galaxy_table['eF395_auto'][galaxy_index],
                                 galaxy_table['eF410_auto'][galaxy_index],
                                 galaxy_table['eF430_auto'][galaxy_index],
                                 galaxy_table['eF515_auto'][galaxy_index],
                                 galaxy_table['eF660_auto'][galaxy_index],
                                 galaxy_table['eF861_auto'][galaxy_index],
                                 galaxy_table['euJAVA_auto'][galaxy_index],
                                 galaxy_table['eg_auto'][galaxy_index],
                                 galaxy_table['er_auto'][galaxy_index],
                                 galaxy_table['ei_auto'][galaxy_index],
                                 galaxy_table['ez_auto'][galaxy_index]])

    wls = np.array([3772.55925537, 3940.44301183, 4094.05290923, 4291.28342875,
                    5132.43006063, 6613.57355705, 8605.51162257, 3527.31414,
                    4716.26251124, 6222.43591412, 7644.41843457, 8912.95141873])

    f_nu = 3631 * 10 ** (-0.4 * magnitudes)
    f_lamb = (1 / (3.34e4 * (wls ** 2))) * f_nu

    f_error = (magnitude_errors / 1.086) * f_lamb

    return wls, f_lamb, f_error


def plot_model_and_observation(galaxy_table, galaxy_index, age, tau, mass, metallicity, Av):
    """

    Overplots BAGPIPES model and S-PLUS observation

    """

    model = make_bagpipes_model(age=age, tau=tau, mass=mass, metallicity=metallicity, Av=Av,
                                z=galaxy_table['z'][galaxy_index])

    wl, f_lamb, f_error = read_splus_photometry(galaxy_table, galaxy_index)

    z = galaxy_table['z'][galaxy_index]

    wave_limit = (model.wavelengths*(1 + z) > 3000) & (model.wavelengths*(1 + z) < 10000)

    model.sfh.plot()

    plt.figure(figsize=(13, 7))
    plt.plot(model.wavelengths[wave_limit]*(1 + z), model.spectrum_full[wave_limit]/(1 + z), color='darkcyan',
             label='Model Spectrum')
    plt.plot(model.filter_set.eff_wavs, model.photometry/(1 + z), 'o', ms=10, color='navy', label='Model Photometry')
    plt.errorbar(wl, f_lamb, yerr=f_error, fmt='o', ms=10, color='forestgreen', label='Observed Photometry')

    plt.legend(frameon=False, fontsize=14)

    plt.xlabel(r'$\lambda \mathrm{[\AA]}$', fontsize=16)
    plt.ylabel(r'$F_\lambda \mathrm{[erg \, cm^2 \, s^{-1} \, \AA^{-1}]}$', fontsize=16)

    return model


if __name__ == '__main__':
    from astropy.table import Table

    catalog = Table.read('splus_laplata.txt', format='ascii')

    # Select only galaxies with no missing bands and r magnitude < 17
    catalog = catalog[(catalog['nDet_auto'] == 12) & (catalog['r_auto'] < 17) & (catalog['class_2'] == 'GALAXY')]

    # ETG : 10
    # Star-forming :

    model_galaxy = plot_model_and_observation(galaxy_table=catalog, galaxy_index=10, age=6, tau=2, mass=11,
                                              metallicity=1, Av=1)
    plt.show()

