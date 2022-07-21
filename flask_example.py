from flask import Flask, request
from flask_cors import CORS
from decimal import Decimal
from radis import calc_spectrum

import json
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app) # pip install flask-cors


@app.route('/post_json', methods=['POST'])
def process_json():
    # put incoming JSON into a dictionary
    data = json.loads(request.data)

    print(data)

    # verify the information in the dictionary
    __param_check(data)

    # convert wavenumber (cm^-1) to wavelenth (nm)
    data["min_wave"], data["max_wave"] = __convert_wavenum_wavelen(
        data["min_wave"], data["max_wave"])

    print("----- output verified params to console as self-check -----")
    for key, value in data.items():
        print("  %s: %s" % (key, value))

    # perform: transmission spectrum of gas sample (calc_spectrum)
    #      --> blackbody spectrum of source (sPlanck)
    #      --> transmission spectrum of beamsplitter and cell windows
    #      --> detector response spectrum
    print("----- start __generate_spectra() -----")
    result = __generate_spectra(data)
    print("----- end __generate_spectra() -----")

    # convert dictionary values to strings and return as JSON
    return json.dumps(str(result))


# --------------------------------------
# ---- spectra calculation functons ----
# --------------------------------------


def __KBr(data):

    if data == None:
        return False

    for x in data:
        datapoint = (0.92267) / (1 + (25.66477 / (x / 1000))
                                 ** -12.35159) ** 0.17344
        data[x] = datapoint * data[x]

    return data


def __CaF2(data):

    if data == None:
        return False

    for x in data:
        datapoint = (0.93091) / (1 + (11.12929 / (x / 1000))
                                 ** -12.43933) ** 4.32574
        data[x] = datapoint * data[x]

    return data


def __ZnSe(data):

    if data == None:
        return False

    for x in data:
        x_um = x / 1000
        datapoint = (0.71015) / ((1 + (20.99353 / x_um) ** -19.31355) ** 1.44348) + -0.13265 / (2.25051 * math.sqrt(
            math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 16.75) ** 2) / (2.25051 ** 2))
        data[x] = datapoint * data[x]

    return data


def __sapphire(data):

    if data == None:
        return False

    for x in data:
        # Gets accurate graph with numpy float128 but throws runtime overflow error
        # datapoint = Decimal(0.78928) / Decimal(1 + (11.9544 / (x / 1000)) ** -12.07226 ) ** (Decimal(6903.57039))
        datapoint = np.float128(0.78928) / np.float128(1 + (11.9544 /
                                                            (x / 1000)) ** -12.07226) ** (np.float128(6903.57039))
        dp2 = datapoint * np.float128(data[x])
        data[x] = dp2

    return data


def __AR_ZnSe(data):

    if data == None:
        return False

    for x in data:
        x_um = x / 1000
        datapoint = (0.82609) / ((1 + ((34.63971 / x_um) ** -8.56269)) ** 186.34792) + -0.47 / (0.55 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 1.47) ** 2) / (0.55 ** 2)) + -0.03456 / (0.4 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 2.88) ** 2) / (0.4 ** 2)) + -0.009 / (0.3 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 6.16) ** 2) / (0.3 ** 2)) + -0.09 / (1 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 16.2) ** 2) / (1 ** 2)) + -0.08 / \
            (1 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 17.4) ** 2) / (1 ** 2)) + 1.12 / (8 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 9.5) ** 2) / (8 ** 2)) + 0.11546 / (2 * math.sqrt(math.pi / (4 * math.log(2)))) * \
            math.exp(-4 * math.log(2) * ((x_um - 4.9) ** 2) / (2 ** 2)) + 0.21751 / (2 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) *
                                                                                                                                            ((x_um - 2.6) ** 2) / (2 ** 2)) + -0.05 / (0.07 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 0.8) ** 2) / (0.07 ** 2))

        data[x] = datapoint * data[x]

    return data


def __AR_CaF2(data):

    if data == None:
        return False

    for x in data:
        x_um = x / 1000
        datapoint = (0.9795) / ((1 + ((18.77617 / x_um) ** -6.94246)) ** 91.98745) + -0.06 / (0.08 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 0.76) ** 2) / (0.08 ** 2))+-0.06 / (0.2 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * (x_um-1.06) ** 2/0.20 ** 2) + -0.6 / (3.0 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um-4.85) ** 2) / (3.0 ** 2)) + -0.35 / (1.0 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           ((x_um - 9.40) ** 2) / (1.00 ** 2)) + 0.05 / (0.8 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 2.60) ** 2) / (0.8 ** 2)) + 0.04 / (0.5 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 7.75) ** 2) / (0.50 ** 2)) + -0.01 / (0.6 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 6.55) ** 2) / (0.6 ** 2)) + 0.01 / (0.5 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 1.82) ** 2) / (0.5 ** 2))
        data[x] = datapoint * data[x]

    return data


def __InSb(data):

    if data == None:
        return False

    for x in data:
        x_um = x / 1000
        datapoint = 1.97163E11 * (1 / (1 + math.exp(-(x_um - 5.3939) / 1.6624))) * (1 - 1 / (1 + math.exp(-(x_um - 5.3939) / 0.11925))) + (
            3.3e10) / (2.44977 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 5) ** 2) / (2.44977 ** 2))
        data[x] = datapoint * data[x]

    return data


def __MCT(data):

    if data == None:
        return False

    for x in data:
        x_um = x / 1000
        datapoint = (1.98748 * (10 ** 9)) + (2.10252 * (10 ** 10)) * (1 / (1 + math.exp(-(x_um - 20.15819) / 5.73688))) * (1 - 1 / (1 + math.exp(-(x_um - 20.15819) /
                                                                                                                                                 1.11659))) + (1.3 * (10 ** 9)) / (2 * math.sqrt(math.pi / (4 * math.log(2)))) * math.exp(-4 * math.log(2) * ((x_um - 18.6) ** 2) / (2 ** 2))
        data[x] = datapoint * data[x]

    return data


def __sPlanck(spectrum, temp):
    H = 6.62606957e-34
    C = 2.99792458e8
    K_B = 1.3806488e-23

    if spectrum == None:
        return False

    for x in spectrum:
        x2 = x * (10 ** -9)
        p = ((0.2 * H * (C ** 2)) / ((x2 ** 4) * x)) * \
            (1 / (math.exp((H * C) / (x2 * K_B * temp)) - 1))
        spectrum[x] = spectrum[x] * p

    return spectrum

# -------------------------------------
# ---------- helper functons ----------
# -------------------------------------


def __error(error_text):
    print(error_text)
    quit()


def __loadData(s):

    data = {}

    for key, val in zip(s[0], s[1]):
        data[float(key)] = float(val)

    return data


def __param_check(data):
    # print("----- list all keys -----")
    # print("  %s" % (data.keys()))

    print("----- check if params are correct -----")
    valid_params = ["min_wave",
                    "max_wave",
                    "molecule",
                    "pressure",
                    "resolution",
                    "num_scan",
                    "zero_fill",
                    "source",
                    "beamsplitter",
                    "cell_window",
                    "detector"]
    for key, value in data.items():
        if key in valid_params:
            if (data[key] == "") or (data[key] == None):
                print("  error with key: %s. Value is: %s" % (key, value))
                return "Error with key: %s. Value is: %s" % (key, value)
            else:
                print("  %s: %s" % (key, value))
        else:
            print("  error with key: %s. Value is: %s" % (key, value))
            return "Error with key: %s. Value is: %s" % (key, value)

    print("----- check if wavenumbers are correct -----")
    if 12500 < data["min_wave"] < 400:
        __error(
            "  wavenumber is out of range (400 - 12500). provided min: %s. provided max: %s" % (data["min_wave"], data["max_wave"]))
    elif data["min_wave"] > data["max_wave"]:
        __error("  min wavenumber is greater than max wavenumber. provided min: %s  provided max: %s" % (
            data["min_wave"], data["max_wave"]))
    elif data["max_wave"] < data["min_wave"]:
        __error("  max wavenumber is less than min wavenumber. provided min: %s  provided max: %s" % (
            data["min_wave"], data["max_wave"]))
    elif data["min_wave"] == data["max_wave"]:
        __error("  min wavenumber is equivalent to max wavenumber. provided min: %s  provided max: %s" % (
            data["min_wave"], data["max_wave"]))
    else:
        print("  good!")

    print("----- check if the molecule is correct -----")
    valid_molecules = ["C2H2",
                       "C2H4",
                       "C2H6",
                       "C2N2",
                       "C4H2",
                       "CF4",
                       "CH3Br",
                       "CH3Cl",
                       "CH3CN",
                       "CH3OH",
                       "CH4",
                       "ClO",
                       "ClONO2",
                       "CO",
                       "CO2",
                       "COCl2",
                       "COF2",
                       "CS",
                       "H2",
                       "H2CO",
                       "H2O",
                       "H2O2",
                       "H2S",
                       "HBr",
                       "HC3N",
                       "HCl",
                       "HCN",
                       "HCOOH",
                       "HF",
                       "HI",
                       "HNO3",
                       "HO2",
                       "HOBr",
                       "HOCl",
                       "N2",
                       "N2O",
                       "NH3",
                       "NO",
                       "NO+",
                       "NO2",
                       "O",
                       "O2",
                       "O3",
                       "OCS",
                       "OH",
                       "PH3",
                       "SF6",
                       "SO2",
                       "SO3"]
    if data["molecule"] in valid_molecules:
        print("  good!")
    else:
        __error("  molecule is not valid. provided molecule: %s") % (
            data["molecule"])

    print("----- check if the pressure is correct -----")
    if 0.0001 <= data["pressure"] <= 10:
        print("  good!")
    else:
        __error(
            "  pressure is out of range (0.0001 - 10). provided pressure: %s") % (data["pressure"])

    print("----- check if the resolution is correct -----")
    valid_resolution = [1, 0.5, 0.25, 0.125, 0.0625]
    if data["resolution"] in valid_resolution:
        print("  good!")
    else:
        __error("  resolution is not valid. provided resolution: %s" %
                (data["resolution"]))

    print("----- check if the number of scans is correct -----")
    if (1 >= data["num_scan"] <= 1024):
        print("  good!")
    else:
        __error(
            "  number of scans is out of range (1 - 1024). provided number of scans: %s") % (data("num_scan"))

    print("----- check if the zero fill is correct -----")
    valid_fill = [0, 1, 2]
    if data["zero_fill"] in valid_fill:
        print("  good!")
    else:
        __error("  zero fill is not valid. provided zero fill: %s" %
                (data["zero_fill"]))

    print("----- check if source is correct -----")
    if data["source"] == "globar":
        data["source"] = 1700
        print("  good!")
    elif data["source"] == "tungsten":
        data["source"] = 3100
        print("  good!")
    else:
        __error("  source is not valid. provided source: %s" %
                (data["source"]))

    print("----- check if beamsplitter is correct -----")
    if (data["beamsplitter"] == "AR_CaF2") or (data["beamsplitter"] == "AR_ZnSe"):
        print("  good!")
    else:
        __error("  beamsplitter is not valid. provided beamsplitter: %s" %
                (data["beamsplitter"]))

    print("----- check if cell window is correct -----")
    if (data["cell_window"] == "ZnSe") or (data["cell_window"] == "CaF2"):
        print("  good!")
    else:
        __error("  cell window is not valid. provided cell window: %s" %
                (data["cell_window"]))

    print("----- check if detector is correct -----")
    if (data["detector"] == "InSb") or (data["detector"] == "MCT"):
        print("  good!")
    else:
        __error("  detector is not valid. provided detector: %s" %
                (data["detector"]))


def __convert_wavenum_wavelen(min, max):
    min_wave = math.floor(10000000 / max)
    max_wave = math.ceil(10000000 / min)

    # check if wavelengths are correct
    if 25000 < min_wave < 800:
        __error(
            "  wavelength is out of range (800 - 25000). provided min: %s. provided max: %s" % (min_wave, max_wave))
    elif min_wave > max_wave:
        __error("  min wavelength is greater than max wavelength. provided min: %s  provided max: %s" % (
            min_wave, max_wave))
    elif max_wave < min_wave:
        __error("  max wavelength is less than min wavelength. provided min: %s  provided max: %s" % (
            min_wave, max_wave))
    elif min_wave == max_wave:
        __error("  min wavelength is equivalent to max wavelength. provided min: %s  provided max: %s" % (
            min_wave, max_wave))
    else:
        return min_wave, max_wave


def __generate_spectra(data):
    # ----- a.) transmission spectrum of gas sample -----
    # https://radis.readthedocs.io/en/latest/source/radis.lbl.calc.html#radis.lbl.calc.calc_spectrum
    s = calc_spectrum(data["min_wave"],
                      data["max_wave"],
                      wunit='nm',
                      molecule=data["molecule"],
                      isotope='1,2,3',
                      pressure=data["pressure"],
                      Tgas=294.15,       # hardecode
                      path_length=10,    # hardcode
                      wstep=0.5,         # (cm^-1)
                      verbose=False,     # hides HITRAN output
                      databank='hitran',
                      warnings={'AccuracyError': 'ignore'},
                      )

    # ----- b.) blackbody spectrum of source -----
    spectrum = __loadData(
        s.get('radiance_noslit', wunit='nm', Iunit='W/cm2/sr/nm'))

    spectrum = __sPlanck(spectrum, data["source"])

    # ----- c.) transmission spectrum of windows/beamsplitter -----

    # Beamsplitter
    if data["beamsplitter"] == "AR_ZnSe":
        spectrum = __AR_ZnSe(spectrum)
    elif data["beamsplitter"] == "AR_CaF2":
        spectrum = __AR_CaF2(spectrum)

    # Cell Windows
    if data["cell_window"] == "CaF2":
        spectrum = __CaF2(spectrum)
        spectrum = __CaF2(spectrum)
    elif data["cell_window"] == "ZnSe":
        spectrum = __ZnSe(spectrum)
        spectrum = __ZnSe(spectrum)

    # ----- d.) detector response spectrum -----

    if data["detector"] == "MCT":
        spectrum = __ZnSe(spectrum)
        spectrum = __MCT(spectrum)
    elif data["detector"] == "InSb":
        spectrum = __sapphire(spectrum)
        spectrum = __InSb(spectrum)

    return spectrum


if __name__ == '__main__':
    app.run()
