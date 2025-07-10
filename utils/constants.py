"""Constants and reference values for geochemical calculations"""

# Chondrite normalization values (McDonough & Sun, 1995)
CHONDRITE_VALUES = {
    'La': 0.237,
    'Ce': 0.613,
    'Pr': 0.0928,
    'Nd': 0.457,
    'Sm': 0.148,
    'Eu': 0.0563,
    'Gd': 0.199,
    'Tb': 0.0361,
    'Dy': 0.246,
    'Ho': 0.0546,
    'Er': 0.160,
    'Tm': 0.0247,
    'Yb': 0.161,
    'Lu': 0.0246
}

# Primitive mantle normalization values (McDonough & Sun, 1995)
PRIMITIVE_MANTLE_VALUES = {
    'Ba': 6.6,
    'Th': 0.085,
    'U': 0.021,
    'Nb': 0.71,
    'Ta': 0.041,
    'La': 0.687,
    'Ce': 1.775,
    'Pr': 0.276,
    'Nd': 1.354,
    'Sr': 21.1,
    'Sm': 0.444,
    'Zr': 11.2,
    'Hf': 0.309,
    'Eu': 0.168,
    'Gd': 0.596,
    'Tb': 0.108,
    'Dy': 0.737,
    'Ho': 0.164,
    'Er': 0.480,
    'Tm': 0.074,
    'Yb': 0.493,
    'Lu': 0.074,
    'Y': 4.55,
    'Pb': 0.071,
    'Rb': 0.635,
    'Cs': 0.032,
    'K': 250,
    'Ti': 1205,
    'P': 95,
    'Sc': 16.9,
    'V': 132,
    'Cr': 2625,
    'Mn': 1045,
    'Co': 105,
    'Ni': 1960,
    'Cu': 30,
    'Zn': 55
}

# Isotope decay constants
LAMBDA_147SM = 6.54e-12  # yr^-1
LAMBDA_176LU = 1.867e-11  # yr^-1
LAMBDA_87RB = 1.42e-11  # yr^-1

# Present-day isotope ratios
CHUR_143ND_144ND = 0.512638
CHUR_176HF_177HF = 0.282785
BULK_EARTH_87SR_86SR = 0.7047

# Depleted mantle values
DM_143ND_144ND = 0.51315
DM_176HF_177HF = 0.28325

# Atomic masses
ATOMIC_MASSES = {
    'Lu': 174.9668,
    'Hf': 178.49,
    'Sm': 150.36,
    'Nd': 144.242,
    'Rb': 85.4678,
    'Sr': 87.62
}

# Isotope abundances
ISOTOPE_ABUNDANCES = {
    'Lu_176': 0.02599,
    'Hf_177': 0.18606,
    'Sm_147': 0.1499,
    'Nd_144': 0.2383,
    'Rb_87': 0.2783,
    'Sr_86': 0.0986
}

# Molecular weights for major elements
MOLECULAR_WEIGHTS = {
    'SiO2': 60.0843,
    'TiO2': 79.8988,
    'Al2O3': 101.9613,
    'FeO': 71.8464,
    'MnO': 70.9375,
    'MgO': 40.3044,
    'CaO': 56.0794,
    'Na2O': 61.9789,
    'K2O': 94.1960,
    'P2O5': 141.9445
}

# Common rock classification fields
ROCK_TYPES = [
    'Basalt', 'Andesite', 'Dacite', 'Rhyolite',
    'Gabbro', 'Diorite', 'Tonalite', 'Granodiorite', 'Granite',
    'Peridotite', 'Pyroxenite', 'Dunite', 'Harzburgite', 'Lherzolite',
    'Eclogite', 'Amphibolite', 'Granulite', 'Schist', 'Gneiss'
]

# Tectonic settings
TECTONIC_SETTINGS = [
    'Mid-ocean ridge', 'Ocean island', 'Volcanic arc', 'Back-arc',
    'Continental rift', 'Continental arc', 'Collision zone',
    'Intraplate', 'Transform fault'
]
