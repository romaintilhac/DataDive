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

# Decay constants (in yr⁻¹) - Updated with more accurate values
LAMBDA_147SM = 6.539e-12  # Begemann et al. 2001
LAMBDA_176LU = 1.865e-11  # Scherer et al. 2001
LAMBDA_87RB = 1.42e-11   # yr^-1

# CHUR values (Bouvier et al. 2008)
CHUR_147SM_144ND = 0.196
CHUR_176LU_177HF = 0.034
CHUR_143ND_144ND = 0.512630
CHUR_176HF_177HF = 0.282785

# Depleted mantle values
DM_147SM_144ND = 0.214
DM_143ND_144ND = 0.513215
DM_176LU_177HF = 0.038  # Griffin et al. 2000
DM_176HF_177HF = 0.283251  # Griffin et al. 2000

# DM Vervoort & Kemp 2025
DM_176LU_177HF_VK25 = 0.04052
DM_176HF_177HF_VK25 = 0.283251

# Hf-Nd mantle array parameters
HFND_ARRAY_A = 1.3432
HFND_ARRAY_B = 2.5348

# Natural abundances
ABUNDANCE_147SM = 0.1499
ABUNDANCE_144ND = 0.238
ABUNDANCE_176LU = 0.0259
ABUNDANCE_177HF = 0.186

# Bulk Earth values
BULK_EARTH_87SR_86SR = 0.7047

# Atomic masses (g/mol)
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

# Column ordering for data presentation
META_ORDER = "Sample Lithology Unit Zone Distance Lat(N) Long(E) dup# run#".split()
ME_ORDER = "SiO2 TiO2 Al2O3 Cr2O3 Fe2O3 Fe2O3T FeO FeOT NiO MnO MgO CaO Na2O K2O P2O5 LOI Total".split()
ME_RATIO_ORDER = "Mg#".split()
TE_ORDER = (
    "La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "F Cl H Ag As Au B Ba Be Bi Br C Ca Cd Co Cr Cs Cu Ga Ge Hf In "
    "Ir K Li Mn Mo N Na Nb Ni Os P Pb Pd Pt Rb Re Rh Ru Sb Sc Se Sn Sr Ta Te Th Ti Tl U V W Y Zn Zr"
).split()
ISO_ORDER = (
    "87Rb/86Sr 87Sr/86Sr 87Sr/86Sr(i) "
    "147Sm/144Nd 143Nd/144Nd 143Nd/144Nd(i) εNd εNd(i) "
    "176Lu/177Hf 176Hf/177Hf 176Hf/177Hf(i) εHf εHf(i) ΔεHf ΔεHf(i) "
    "208Pb/204Pb 207Pb/204Pb 206Pb/204Pb"
).split()

# Plot labels with LaTeX formatting
MANUAL_LABELS = {
    'Distance': "Distance from Moho [m]",
    "εNd": r"$\mathrm{\varepsilon}_{\mathrm{Nd}}$",
    "εHf": r"$\mathrm{\varepsilon}_{\mathrm{Hf}}$",
    "εNd(i)": r"$\mathrm{\varepsilon}_{\mathrm{Nd}}(i)$",
    "εHf(i)": r"$\mathrm{\varepsilon}_{\mathrm{Hf}}(i)$",
    "ΔεHf": r"$\Delta \mathrm{\varepsilon}_{\mathrm{Hf}}$",
    "ΔεHf(i)": r"$\Delta \mathrm{\varepsilon}_{\mathrm{Hf}}(i)$",
}

# Molecular weights for major elements (g/mol)
MOLECULAR_WEIGHTS = {
    'SiO2': 60.0848,
    'TiO2': 79.8988,
    'Al2O3': 101.96128,
    'Cr2O3': 151.9902,
    'FeO': 71.8464,
    'Fe2O3': 159.6922,
    'MnO': 70.9374,
    'MgO': 40.3044,
    'NiO': 74.7094,
    'ZnO': 81.3794,
    'CaO': 56.0794,
    'Na2O': 61.97894,
    'K2O': 94.1954,
    'P2O5': 141.94452,
    'V2O5': 181.8798
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
