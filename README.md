# Catégorisation des produits d'un boutique en ligne

Fourni une API permettant de prédire la catégorie d'un produit en fonction de sa désignation, description et d'une image.

## Structure du projet

E-COMMERCE-MLOPS/  
├── .github/workflows : Github Actions  
├── backend/ : API et modèle de prédiction  
├── frontend/ : (plus tard) application WEB pour interagir avec l'API  
└── scripts/ : liste des scripts permettant de lancer les tests ou de faire le déploiement  

## Backend

### Configuration de l'environnement de développement

Un conteneur de développement a été préparé.  
Voici les prérequis : 
- [Visual Studio Code](https://code.visualstudio.com) 
- Extension VSCode [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)
- Extension VSCode [Dev Container](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

Une fois les extensions installées, veuillez-suivre les étapes suivantes :  
1. Ouvrir le projet dans VSCode.
2. Cloner le repo e-commerce-mlops sur le terminal et placer vous dans le dossier. 
3. Ouvrir la palette des commandes (Cmd+Shift+p).
4. Saisir **dev container open** et sélectionner l'option **Dev Containers: Open folder in Container...**.
5. Sélectionner le dossier **backend**.

La fenêtre de VSCode va se rouvrir. Dans le terminal, vous pourrez constater l'installation des packages.

### Lancement de l'API

L'API utilise **uvicorn** comme serveur Web.  
Pour démarrer le serveur et observer les changements, exécuter la commande suivante dans le terminal du conteneur de développement :

```shell
uvicorn app.main:app --reload
```

VSCode doit faire automatiquement la redirection du port 8000.  
Ouvrir l'addresse ci-dessous dans un navigateur Web sur la machine hôte pour afficher la documentation :

http://127.0.0.1:8000/docs

### Génération d'une nouvelle migration

Lorsque des modifications sont réalisées sur la base de données, il est nécessaire de générer un nouveau script de migration pour permettre a Alembic de mettre à jour la BDD.

