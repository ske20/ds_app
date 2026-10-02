import math
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt
import streamlit as st

PDFS = ["Normal", "Truncated normal", "Beta", "Lognormal", "Exponential", "Triangular", "Uniform"]

st.set_page_config(
    page_title="Probability Distribution Learning App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Probability Distribution Teaching App")
st.markdown("Explore data distributions, fit candidate models, run Monte Carlo simulations, and visualize uncertainty.")

# Sidebar Controls
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

use_sample = False
if uploaded_file is None:
    st.sidebar.info("No file uploaded. You can load sample data to test the app.")
    if st.sidebar.button("Load Sample Data"):
        use_sample = True

@st.cache_data
def get_sample_data():
    np.random.seed(42)
    data = np.random.normal(loc=50, scale=10, size=500)
    return pd.DataFrame({"Value": data, "Measurement": np.random.exponential(scale=5, size=500)})

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")
        df = None
elif use_sample:
    df = get_sample_data()
else:
    df = None

if df is not None:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found in the dataset.")
        st.stop()
    
    col = st.sidebar.selectbox("Select Numeric Column", numeric_cols)
    
    # Clean data
    x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    
    if len(x) < 5:
        st.error("At least five numeric values are required.")
        st.stop()
        
    st.sidebar.markdown("---")
    st.sidebar.header("2. PDF Fitting Options")
    chosen_pdfs = []
    pvars = {}
    for name in PDFS:
        pvars[name] = st.sidebar.checkbox(name, value=True)
        if pvars[name]:
            chosen_pdfs.append(name)
            
    if not chosen_pdfs:
        st.warning("Please select at least one PDF to fit.")
        st.stop()
        
    bins = st.sidebar.slider("Number of histogram bins", min_value=2, max_value=100, value=10)
    
    @st.cache_data
    def fit_all_distributions(_x, chosen):
        mn = float(_x.min())
        mx = float(_x.max())
        sp = mx - mn
        out = []
        for name in chosen:
            try:
                if name == "Normal":
                    d = stats.norm
                    p = d.fit(_x)
                    k = 2
                elif name == "Truncated normal":
                    d = stats.truncnorm
                    lo = max(0, mn)
                    hi = mx
                    def fun(z):
                        mu, ls = z
                        sd = np.exp(ls)
                        return -np.sum(d.logpdf(_x, (lo - mu) / sd, (hi - mu) / sd, loc=mu, scale=sd))
                    res = optimize.minimize(fun, [_x.mean(), np.log(max(_x.std(ddof=1), 1e-6))], method="Nelder-Mead")
                    mu, ls = res.x
                    sd = np.exp(ls)
                    p = ((lo - mu) / sd, (hi - mu) / sd, mu, sd)
                    k = 2
                elif name == "Beta":
                    if sp <= 0:
                        continue
                    d = stats.beta
                    lo = 0 if mn >= 0 else mn - .05 * sp
                    hi = mx + max(.05 * sp, 1e-6)
                    p = d.fit(_x, floc=lo, fscale=hi - lo)
                    k = 2
                elif name == "Lognormal":
                    if np.any(_x <= 0):
                        continue
                    d = stats.lognorm
                    p = d.fit(_x, floc=0)
                    k = 2
                elif name == "Exponential":
                    d = stats.expon
                    p = d.fit(_x, floc=max(0, mn))
                    k = 1
                elif name == "Triangular":
                    d = stats.triang
                    p = d.fit(_x)
                    k = 3
                else:
                    d = stats.uniform
                    p = (mn, sp)
                    k = 2
                ks, pv = stats.kstest(_x, d.cdf, args=p)
                ll = np.sum(d.logpdf(_x, *p))
                out.append(dict(name=name, dist=d, params=p, ks=ks, pv=pv, aic=2 * k - 2 * ll, bic=k * math.log(len(_x)) - 2 * ll))
            except Exception:
                pass
        return out

    fits = fit_all_distributions(x, tuple(chosen_pdfs))
    
    if not fits:
        st.error("No selected PDF could be fitted successfully.")
        st.stop()
        
    fit_names = [z["name"] for z in fits]
    
    st.sidebar.markdown("---")
    st.sidebar.header("3. Monte Carlo Settings")
    selected_pdf_name = st.sidebar.selectbox("PDF used for Monte Carlo", fit_names)
    nsim = st.sidebar.number_input("Accepted draws", min_value=1000, max_value=1000000, value=10000, step=1000)
    
    current_fit = next(z for z in fits if z["name"] == selected_pdf_name)
    
    @st.cache_data
    def run_sim(fit_dict, n_draws, x_min, x_max):
        lo = max(0, float(x_min))
        hi = float(x_max)
        rng = np.random.default_rng(557)
        parts = []
        got = 0
        while got < n_draws:
            z = np.asarray(fit_dict["dist"].rvs(*fit_dict["params"], size=max(5000, n_draws - got), random_state=rng))
            z = z[np.isfinite(z) & (z >= lo) & (z <= hi)]
            parts.append(z)
            got += len(z)
        mc_data = np.concatenate(parts)[:n_draws]
        q_vals = np.percentile(mc_data, [10, 50, 90])
        return mc_data, q_vals

    mc, q = run_sim(current_fit, int(nsim), x.min(), x.max())
    
    def gx():
        sp = np.ptp(x) or 1
        return np.linspace(max(0, x.min() - .1 * sp), x.max() + .1 * sp, 700)

    st.sidebar.markdown("---")
    st.sidebar.header("4. Display Settings")
    hist_color = st.sidebar.color_picker("Histogram Color", "#5B7BEA")
    line_color = st.sidebar.color_picker("Fit Line Color", "#C00070")

    # Status summary banner
    st.success(f"**Column:** {col} | **n:** {len(x)} | **Selected PDF:** {current_fit['name']} | **KS Statistic:** {current_fit['ks']:.4f}")

    # View selection
    view_option = st.selectbox(
        "Choose Display View",
        [
            "1. Raw data",
            "2. Frequency histogram",
            "3. Descriptive statistics",
            "4. Empirical probability + fit",
            "5. Empirical PDF",
            "6. Candidate PDFs",
            "7. Empirical + fitted CDFs",
            "8. Goodness-of-fit comparison",
            "9. Monte Carlo simulation",
            "10. Uncertainty percentiles",
            "11. Full teaching dashboard"
        ]
    )

    if view_option.startswith("1."):
        st.subheader("Raw Data")
        raw_df = pd.DataFrame({"Observation": np.arange(1, len(x) + 1), col: x})
        st.dataframe(raw_df, use_container_width=True)

    elif view_option.startswith("2."):
        st.subheader("Frequency Histogram")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(x, bins=bins, color=hist_color, edgecolor="black")
        ax.set(title="Frequency histogram", xlabel=col, ylabel="Frequency")
        fig.tight_layout()
        st.pyplot(fig)

    elif view_option.startswith("3."):
        st.subheader("Descriptive Statistics")
        desc_df = pd.DataFrame({
            "Statistic": ["Count", "Minimum", "Q1", "Median", "Mean", "Q3", "Maximum", "IQR", "Sample SD", "Skewness"],
            "Value": [len(x), x.min(), np.percentile(x, 25), np.median(x), x.mean(), np.percentile(x, 75), x.max(), stats.iqr(x), x.std(ddof=1), stats.skew(x, bias=False)]
        })
        st.dataframe(desc_df, use_container_width=True)

    elif view_option.startswith("4."):
        st.subheader("Empirical Probability Histogram + Selected Fit")
        fig, ax = plt.subplots(figsize=(10, 6))
        c, e = np.histogram(x, bins=bins)
        pr = c / len(x)
        m = (e[:-1] + e[1:]) / 2
        ax.bar(e[:-1], pr, width=np.diff(e), align="edge", color=hist_color, edgecolor="black", alpha=.65, label="Empirical")
        ax.plot(m, np.diff(current_fit["dist"].cdf(e, *current_fit["params"])), "o-", color=line_color, lw=2, label=current_fit["name"])
        for xx, y, n_val in zip(m, pr, c):
            if n_val > 0 and y > 0:
                ax.text(xx, y, f"{y*100:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set(title="Empirical probability histogram + selected fit", xlabel=col, ylabel="Probability within bin")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    elif view_option.startswith("5."):
        st.subheader("Empirical Density and Selected PDF")
        fig, ax = plt.subplots(figsize=(10, 6))
        g = gx()
        ax.hist(x, bins=bins, density=True, color=hist_color, edgecolor="black", alpha=.4, label="Empirical density histogram")
        ax.plot(g, stats.gaussian_kde(x)(g), "--", color="#333333", label="KDE")
        ax.plot(g, current_fit["dist"].pdf(g, *current_fit["params"]), color=line_color, lw=2.5, label=f'Selected {current_fit["name"]} PDF')
        ax.set(title="Empirical density and selected PDF", xlabel=col, ylabel="Probability density")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    elif view_option.startswith("6."):
        st.subheader("Candidate PDFs Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        g = gx()
        ax.hist(x, bins=bins, density=True, color=hist_color, edgecolor="black", alpha=.25)
        for z in fits:
            is_sel = (z["name"] == current_fit["name"])
            ax.plot(g, z["dist"].pdf(g, *z["params"]), color=line_color if is_sel else None, lw=2.5 if is_sel else 1.3, label=f'{z["name"]} KS={z["ks"]:.3f}')
        ax.set(title="Candidate PDFs", xlabel=col, ylabel="Density")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    elif view_option.startswith("7."):
        st.subheader("Empirical and Fitted CDFs")
        fig, ax = plt.subplots(figsize=(10, 6))
        g = gx()
        s = np.sort(x)
        ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", label="Empirical CDF")
        for z in fits:
            is_sel = (z["name"] == current_fit["name"])
            ax.plot(g, z["dist"].cdf(g, *z["params"]), color=line_color if is_sel else None, lw=2.5 if is_sel else 1.3, label=z["name"])
        ax.set(title="Empirical and fitted CDFs", xlabel=col, ylabel="P(X <= x)")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    elif view_option.startswith("8."):
        st.subheader("Goodness-of-Fit Comparison")
        gof_df = pd.DataFrame([{
            "Selected": "Yes" if z["name"] == current_fit["name"] else "No",
            "Distribution": z["name"],
            "KS statistic": z["ks"],
            "KS p-value": z["pv"],
            "AIC": z["aic"],
            "BIC": z["bic"]
        } for z in fits]).sort_values("KS statistic")
        st.dataframe(gof_df, use_container_width=True)

    elif view_option.startswith("9."):
        st.subheader("Monte Carlo Simulation")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(mc, bins=40, density=True, color=hist_color, edgecolor="black", alpha=.65)
        for val, label_str, color_str in zip(q, ["p10", "p50", "p90"], ["green", "orange", "red"]):
            ax.axvline(val, color=color_str, label=f"{label_str}={val:.4g}")
        ax.set(title=f'Monte Carlo using {current_fit["name"]}', xlabel=col, ylabel="Simulated density")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    elif view_option.startswith("10."):
        st.subheader("Uncertainty Percentiles")
        perc_df = pd.DataFrame({
            "Selected PDF": [current_fit["name"]] * 3,
            "Percentile": ["p10", "p50", "p90"],
            "Value": q
        })
        st.dataframe(perc_df, use_container_width=True)

    elif view_option.startswith("11."):
        st.subheader("Full Teaching Dashboard")
        fig, aa = plt.subplots(2, 2, figsize=(12, 10))
        g = gx()
        c, e = np.histogram(x, bins=bins)
        pr = c / len(x)
        m = (e[:-1] + e[1:]) / 2
        
        aa[0, 0].bar(e[:-1], pr, width=np.diff(e), align="edge", color=hist_color, edgecolor="black", alpha=.65)
        aa[0, 0].plot(m, np.diff(current_fit["dist"].cdf(e, *current_fit["params"])), "o-", color=line_color)
        aa[0, 0].set_title("Empirical probability + fit")
        
        aa[0, 1].hist(x, bins=bins, density=True, color=hist_color, edgecolor="black", alpha=.35)
        aa[0, 1].plot(g, current_fit["dist"].pdf(g, *current_fit["params"]), color=line_color)
        aa[0, 1].set_title("Selected PDF")
        
        s = np.sort(x)
        aa[1, 0].step(s, np.arange(1, len(s) + 1) / len(s), where="post")
        aa[1, 0].plot(g, current_fit["dist"].cdf(g, *current_fit["params"]), color=line_color)
        aa[1, 0].set_title("CDF")
        
        aa[1, 1].hist(mc, bins=35, density=True, color=hist_color, edgecolor="black", alpha=.65)
        for val, label_str, color_str in zip(q, ["p10", "p50", "p90"], ["green", "orange", "red"]):
            aa[1, 1].axvline(val, color=color_str, label=f"{label_str}={val:.3g}")
        aa[1, 1].legend()
        aa[1, 1].set_title(f'Monte Carlo: {current_fit["name"]}')
        
        fig.tight_layout()
        st.pyplot(fig)

else:
    st.info("👈 Please upload a CSV file using the sidebar or click **'Load Sample Data'** to begin.")
