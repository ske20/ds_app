# Probability Distribution Teaching App (Streamlit)

An interactive web application built with [Streamlit](https://streamlit.io/) for exploring probability distributions, fitting statistical models, comparing goodness-of-fit metrics (KS, AIC, BIC), running Monte Carlo simulations, and analyzing uncertainty percentiles.

## Features
* **Data Input:** Upload your own CSV dataset or load built-0in sample data instantly.
* **Distribution Fitting:** Fit Normal, Truncated Normal, Beta, Lognormal, Exponential, Triangular, and Uniform distributions.
* **Visualizations:** Frequency histograms, empirical probability histograms, density curves (KDE + PDFs), and CDF step plots.
* **Goodness-of-Fit:** Compare distributions using Kolmogorov-Smirnov statistics, p-values, AIC, and BIC.
* **Monte Carlo Simulation:** Run simulations based on the selected fitted PDF with configurable draws and p10/p50/p90 uncertainty percentiles.
* **Interactive Dashboard:** Full multi-panel teaching dashboard view.

## Local Installation & Running

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```
