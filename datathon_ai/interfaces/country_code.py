import os
from dataclasses import dataclass
from typing import List
import pandas as pd 
from marshmallow import Schema, fields, post_load


@dataclass
class CountryCode:
    id: int
    name: str
    alpha2: str
    alpha3: str


class CountryCodeSchema(Schema):
    id = fields.Integer()
    name = fields.String()
    alpha2 = fields.String()
    alpha3 = fields.String()
    
    @post_load
    def make_country_code(self, data, **kwargs) -> CountryCode:
        return CountryCode(**data)


@dataclass
class CountryReferential:
    """
    Interface that represents the countries_code.csv file
    """
    countries_codes: List[CountryCode]
    
    @classmethod
    def from_csv(cls, path_to_file: str):
        assert os.path.exists(path_to_file)
        df_referential = pd.read_csv(path_to_file)
        referential_as_dict = []
        for _, row in df_referential.iterrows():
            referential_as_dict.append({
                "id": row["id"],
                "name": row["name"],
                "alpha2": row["alpha2"],
                "alpha3": row["alpha3"],
            })
        return cls(countries_codes=CountryCodeSchema(many=True).load(referential_as_dict))
    