import re

# Dictionary for manual label overrides
manual_labels = {
    'Distance': r"Distance from Moho [m]",
    "εNd": r"$\mathrm{\varepsilon}_{\mathrm{Nd}}$",
    "εHf(i)": r"$\mathrm{\varepsilon}_{\mathrm{Hf}}(i)$",
    "ΔεHf": r"$\Delta \mathrm{\varepsilon}_{\mathrm{Hf}}$",
    "ΔεHf(i)": r"$\Delta \mathrm{\varepsilon}_{\mathrm{Hf}}(i)$",
}

def plot_labels(label):
    # First, check if label is manually defined
    if label in manual_labels:
        return manual_labels[label]
    
    # Otherwise apply regex-based formatting
    def repl(match):
        num = match.group(1)
        elem = match.group(2)
        return f"$^{{{num}}}$" + elem

    return re.sub(r'(\d+)([A-Za-z]+)', repl, label)