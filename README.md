Script explanations:
1) Gausssian_fits.py: Using the provided calibration files, makes a 4x4 plot of each strip and via a gaussian function, fits the peaks. From there it exports the values needed for calibration to a separate file.
2) Calibrated_tripple_alpha.py: Using the calibration values exported from gaussian_fits.py, it applies the calibration and builds a 2D plot, showing the calibrated file.
3) Spectrum_calibrated: It is in the name - applied the calibration per strip and shows the sum calibrated spectrum. 
4) Correlation_spectrum+chains.py: Apllying a correlation gates on energy, time and position it plots the distribution of the position difference ∆position between correlated mother and daughter events against the event count. It is then fittet with a gaussian funtion and extracts the sigma, which later is used for position tolerance.
5) :After getting the pos. tolerance we apply the gates and get a correlation plot and the correlation chains printed in the terminal.
6) Lifetime.py: Builds a time-difference (∆t) distribution for correlated α-decays used to extract the lifetime by fitting it with exp. function. Prints the lifetime value.
   
