"""
Enhanced plot formatting utilities with improved label formatting and styling.
"""

import re
from utils.constants import MANUAL_LABELS, TE_ORDER, ME_ORDER

def oxide_to_subscript(formula: str) -> str:
    """Convert oxide formulas to subscript notation"""
    # Mapping digits to unicode subscripts
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return formula.translate(subscript_map)

def format_subscripts(label):
    """Insert LaTeX-style subscripts for numbers"""
    def repl(match):
        elem = match.group(1)
        num = match.group(2)
        return f"{elem}$_{{{num}}}$"
    return re.sub(r'([A-Za-z]+)(\d+)', repl, label)

def format_isotope_labels(label):
    """Format isotope labels with proper superscript notation"""
    def repl(match):
        num = match.group(1)
        elem = match.group(2)
        return f"$^{{{num}}}$" + elem
    return re.sub(r'(\d+)([A-Za-z]+)', repl, label)

def plot_labels(label):
    """
    Generate properly formatted labels for plots.
    Handles manual labels, trace elements, major elements, and isotopes.
    """
    # First, check if label is manually defined
    if label in MANUAL_LABELS:
        return MANUAL_LABELS[label]
    
    # Trace elements: add μg/g units
    if label in TE_ORDER:
        return f"{label} [μg/g]"
    
    # Major elements: format subscripts + add wt. % units
    if label in ME_ORDER:
        return format_subscripts(label) + " [wt. %]"
    
    # Handle isotope ratios
    if '/' in label and any(char.isdigit() for char in label):
        return format_isotope_labels(label)
    
    # Handle epsilon notation
    if label.startswith('ε'):
        return format_subscripts(label)
    
    # Handle delta notation
    if label.startswith('Δ'):
        return format_subscripts(label)
    
    # Default: just format subscripts if any
    return format_subscripts(label)

def format_plot_title(title):
    """Format plot titles with proper notation"""
    return format_subscripts(title)

def get_element_color_map():
    """Get consistent color mapping for elements"""
    return {
        'La': '#1f77b4',
        'Ce': '#ff7f0e',
        'Pr': '#2ca02c',
        'Nd': '#d62728',
        'Sm': '#9467bd',
        'Eu': '#8c564b',
        'Gd': '#e377c2',
        'Tb': '#7f7f7f',
        'Dy': '#bcbd22',
        'Ho': '#17becf',
        'Er': '#aec7e8',
        'Tm': '#ffbb78',
        'Yb': '#98df8a',
        'Lu': '#ff9896'
    }

def get_lithology_color_map():
    """Get consistent color mapping for lithologies"""
    return {
        'Peridotite': '#228B22',
        'Pyroxenite': '#8B4513',
        'Gabbro': '#4169E1',
        'Basalt': '#DC143C',
        'Andesite': '#FF6347',
        'Dacite': '#DAA520',
        'Rhyolite': '#FF1493',
        'Granite': '#CD853F',
        'Diorite': '#708090',
        'Syenite': '#DDA0DD',
        'Unknown': '#808080'
    }

def format_spider_diagram_elements():
    """Get ordered list of elements for spider diagrams"""
    return ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']

def format_multi_element_diagram_elements():
    """Get ordered list of elements for multi-element diagrams"""
    return ['Ba', 'Th', 'U', 'Nb', 'Ta', 'La', 'Ce', 'Pr', 'Nd', 'Sr', 'Sm', 'Zr', 'Hf', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y']