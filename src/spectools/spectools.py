
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, k
from scipy.optimize import curve_fit

file = 'arcturus.txt'

def load_spectrum(file, max_wavelength=15000):
    
    wavelength = []
    flux = []
    # The wl of the spectrum is in the first row.
    # The flux is in the second row. 
    # We will read the file line by line and split each line into two values.
    with open(file, 'r') as file:
        for line in file:
            # Process data lines (wavelength and flux values)
            values = line.split()
            if len(values) == 2:  # Ensure there are exactly two values
                wavelength_value = float(values[0])
                flux_value = float(values[1])
                # Filter by max wavelength
                if wavelength_value <= max_wavelength:
                    wavelength.append(wavelength_value)
                    flux.append(flux_value)
    # Convert lists to numpy arrays for easy manipulation
    wavelength = np.array(wavelength)
    flux = np.array(flux)
    
    return wavelength, flux


def determine_continuum(flux, method='running_average', window_size=50, threshold=0.8):
    # Convert flux to log space for fitting
    log_flux = np.log(flux)

    if method == 'running_average':
        # Initialize an empty continuum array
        continuum = np.zeros_like(flux)
        
        for i in range(len(flux)):
            half_window = min(i, len(flux) - i - 1, window_size // 2)
            
            # Define the window range in linear space
            window_start = i - half_window
            window_end = i + half_window + 1
            
            # Calculate the running average in log-flux space, excluding deep absorption lines
            window_flux = log_flux[window_start:window_end]
            filtered_flux = window_flux[window_flux > np.log(threshold * np.median(np.exp(window_flux)))]
            
            # Take the median of the filtered flux, or the median of the whole window if no filtering
            continuum[i] = np.exp(np.median(filtered_flux)) if len(filtered_flux) > 0 else np.exp(np.median(window_flux))
    else:
        raise ValueError(f"Unknown method: {method}")

    return continuum

def normalize_spectrum(flux, continuum):
    return flux / continuum
    
# Fit Gaussian around each target line
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def calculate_equivalent_width(wavelength, flux):
    spectral_lines = {  
                      'Halpha': 6563,                      
                      'Hbeta': 4861,
                      }
    fwhm_results = []

    # Create the inverted spectrum for fitting
    inverted_spectrum = 1 - flux

    for line_name, center_wavelength in spectral_lines.items():
        # Initial guess based on the target line center
        idx = (np.abs(wavelength - center_wavelength)).argmin()
        amp_guess = inverted_spectrum[idx]
        sigma_guess = 10  # Initial guess for sigma
        guess = [amp_guess, center_wavelength, sigma_guess]

        # Narrow region around the line center for fitting
        mask = (wavelength > center_wavelength - 200) & (wavelength < center_wavelength + 200)

        # Check if there are enough points to fit
        if np.sum(mask) < 5:  # Require at least 5 points in the window
            print(f"Not enough data points to fit Halpha")
        else:   
            popt, _ = curve_fit(gaussian, wavelength[mask], inverted_spectrum[mask], p0=guess)

            # Calculate FWHM from the fit sigma
            amp, mu, sigma = popt
            fwhm = 2 * np.sqrt(2 * np.log(2)) * np.abs(sigma)
            fwhm_results.append((line_name, mu, fwhm, mask, popt))
            print('The FWHM of %s line is %s' % (line_name, fwhm))
    return fwhm_results

def plot_fwhm(wavelength, flux, fwhm_results):
    for line_name, mu, fwhm, mask, popt in fwhm_results:
        # Plot the fitted Gaussian, inverted back to negative values
        plt.plot(wavelength[mask], 1 - gaussian(wavelength[mask], *popt), linestyle='--', label=f'{line_name} FWHM={fwhm:.2f} Å')
        plt.plot(wavelength[mask], flux[mask], label=f'{line_name} Spectrum', color='black')
        plt.title("Spectral Lines with Fitted Gaussians")
        plt.xlabel("Wavelength (Angstrom)")
        plt.ylabel("Normalized Flux")
        plt.legend()
        plt.show()
    


