# Catégorisation des produits d'un boutique en ligne

Fourni une API permettant de prédire la catégorie d'un produit en fonction de sa désignation, description et d'une image.

## Structure du projet

E-COMMERCE-MLOPS/
├── .github/workflows : Github Actions
├── backend/ : API et modèle de prédiction
├── frontend/ : (eventually) application WEB pour interagir avec l'API
└── scripts/ : liste des scripts permettant de lancer les tests ou de faire le déploiement

// Charactères que nous pourrons utiliser pour mettre à jour la structure.
├── 
│   └── 
//

## Mise en place du projet

### Backend

Ce projet utilise Conda comme gestionnaire d'environnement et de paquets.  
Les commandes ci-dessous permettent la création de l'environnement, l'installation de pip et des packages nécessaires au bon fonctionnement du projet.

```shell
conda create --name ecommerce python=3.10.9.final.0
conda activate ecommerce
conda install pip
python -m pip 
```