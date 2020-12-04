import pandas as pd
import numpy as np

def get_country_data():
    countries_key = 'resources/countries_code.csv'
    eu_key = 'resources/eu.csv'

    # Load data into a Pandas Data Frame

    countries = pd.read_csv(countries_key)
    eu = pd.read_csv(eu_key)
    eu = list(np.array(eu.values).transpose()[0])
    names = countries["name"]
    names = list(names) + ["U.S.", "U.K."]
    countries_dict = {}
    for country in countries.values:
        countries_dict[country[0]] = list(country[1:])

    countries_demonym_key = 'resources/countries_demonym.csv'
    countries_demonym = pd.read_csv(countries_demonym_key)
    demonyms = list(countries_demonym["fdemonym"]) + \
        list(countries_demonym["mdemonym"])

    for i in countries_dict.keys():
        country = countries_demonym[countries_demonym["id"] == i]
        countries_dict[i] += list(country["fdemonym"]) + \
            list(country["mdemonym"])
        countries_dict[i] = [str(c).lower() for c in countries_dict[i]]
    return countries_dict, names, eu, demonyms

countries_dict, names, eu, demonyms = get_country_data()

countries_dict_inverse = {}

for key, l in countries_dict.items():
    for x in l:
        countries_dict_inverse[x] = key

