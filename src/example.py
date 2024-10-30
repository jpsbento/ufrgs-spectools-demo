from spectools.spectools import *

filename = 'arcturus.txt'

wavelength, flux = load_spectrum(filename)

continuum = determine_continuum(flux)


plt.figure(figsize=(10, 6))
plt.plot(wavelength, flux, label="Original Spectrum")
plt.plot(wavelength, continuum, label=f"Continuum Fit", linestyle="--")
plt.xlabel("Wavelength (Angstrom)")
plt.ylabel("Flux")
plt.legend()
plt.title("Blackbody Continuum Fit")
plt.show()

normalized_flux = normalize_spectrum(flux, continuum)
