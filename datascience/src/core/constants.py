"""Contains all constants used in the project."""

from pathlib import Path
from typing import Final

import src

ROOT_DIR: Final = Path(Path(src.__file__).parent).parent

CATEGORIES_DIC: Final = {
    10: "Livre",
    1140: "Figurine et produits dérivés",
    1160: "Carte à collectionner",
    1180: "Univers fantastiques",
    1280: "Jouet pour enfant",
    1281: "Jeu de société",
    1300: "Miniature de collection",
    1301: "Loisir",
    1302: "Activité d'extérieur",
    1320: "Accessoire bébé",
    1560: "Meuble d'intérieur",
    1920: "Litterie, rideaux",
    1940: "Epicerie",
    2060: "Décoration d'intérieur",
    2220: "Accessoire animaux de compagnie",
    2280: "Magazine et BD",
    2403: "Livres anciens",
    2462: "Jeu vidéo - Pack",
    2522: "Fourniture de bureau",
    2582: "Meubles extérieur",
    2583: "Piscine",
    2585: "Bricolage",
    2705: "Livre",
    2905: "Jeu vidéo - Jeu",
    40: "Jeu vidéo - Jeu",
    50: "Jeu vidéo - Accessoire",
    60: "Jeu vidéo - Console",
}

CATEGORIES_SIMPLIFIED_DIC: Final = {
    10: 0,
    1140: 1,
    1160: 2,
    1180: 3,
    1280: 4,
    1281: 5,
    1300: 6,
    1301: 7,
    1302: 8,
    1320: 9,
    1560: 10,
    1920: 11,
    1940: 12,
    2060: 13,
    2220: 14,
    2280: 15,
    2403: 16,
    2462: 17,
    2522: 18,
    2582: 19,
    2583: 20,
    2585: 21,
    2705: 22,
    2905: 23,
    40: 24,
    50: 25,
    60: 26,
}

MLFLOW_DEFAULT_MODELS: Final = {
    "text": {
        "name": "text",
        "description": "Product Classification model using text input.",
        "tags": {
            "text_input": "true",
            "type": "tensorflow",
        },
    },
    "image": {
        "name": "image",
        "description": "Product Classification model using image input.",
        "tags": {
            "image_input": "true",
            "type": "tensorflow",
        },
    },
    "fusion": {
        "name": "fusion",
        "description": (
            "Product Classification model using both " "text and image input."
        ),
        "tags": {
            "image_input": "true",
            "text_input": "true",
            "type": "tensorflow",
        },
    },
}
