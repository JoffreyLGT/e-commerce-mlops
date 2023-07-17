# Catégorisation des produits d'un boutique en ligne

Fourni une API permettant de prédire la catégorie d'un produit en fonction de sa désignation, description et d'une image.

## Structure du projet

E-COMMERCE-MLOPS/  
├── .github/workflows : Github Actions  
├── backend/ : API et modèle de prédiction  
├── frontend/ : (plus tard) application WEB pour interagir avec l'API  
└── scripts/ : liste des scripts permettant de lancer les tests ou de faire le déploiement  

## Mise en place du projet

### Backend

Un conteneur de développement a été préparé.  
Voici les prérequis : 
- [Visual Studio Code](https://code.visualstudio.com) 
- Extension VSCode [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)
- Extension VSCode [Dev Container](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

Une fois les extensions installées, veuillez-suivre les étapes suivantes :  
1. Ouvrir le projet dans VSCode.
2. Ouvrir la palette des commandes (Cmd+Shift+p).
3. Saisir **dev container open** et sélectionner l'option **Dev Containers: Open folder in Container...**.
4. Sélectionner le dossier **backend**.

La fenêtre de VSCode va se rouvrir. Dans le terminal, vous pourrez constater l'installation des packages.