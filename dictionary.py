import json

dictionary = {

    # 400
    'min_wave': 1900,

    # 12500
    'max_wave': 2300,

    # like 40 options
    'molecule': 'CO',

    # 10 - 0 (bar)
    'pressure': 1,

    # 1, 0.5, 0.25, 0.125, 0.0625
    'resolution': 1,

    # 1 - 1024
    'num_scan': 1,

    # 0, 1, 2
    'zero_fill': 0,

    # Tungsten or Globar
    'source': 'Tungsten',

    # AR_CaF2 or AR_ZnSe
    'beamsplitter': 'AR_CaF2',

    # ZnSe or CaF2
    'cell_window': 'ZnSe',

    # InSb or MCT
    'detector': 'InSb',

    # 294.15 --> hardcode (K)
    # 'tgas': 294.15,

    # 10 --> hardcode (cm)
    # 'path_length': 10,

    # 1 --> hardcode
    # 'mole_fraction': 1,
}

print("\nprint as dictionary:")
print(dictionary)

print("\nprint as JSON")
print(json.dumps(dictionary, indent=4))
