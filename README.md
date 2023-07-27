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
Ouvrir l’adresse ci-dessous dans un navigateur Web sur la machine hôte pour afficher la documentation :

http://127.0.0.1:8000/docs

### Modification d'une table en BDD

Suivre les étapes de la section [Ajout d'une table en BDD](#ajout-d-une-table-en-bdd).

### Ajout d'une table en BDD

>> TLDR : Ajouter ou modifier les modèles SQLAlchemy dans `/app/models/`, les schemas Pydantic dans `/app/schemas/` et les outils CRUD d'interaction avec la BDD dans `/app/crud/`. Attention à bien mettre à jour les fichier `__init__.py` de ces modules ! Pour terminer, générer la migration alembic.
>> Utiliser **prediction_feedback** comme exemple.

1. Création du modèle SQLAlchemy

Créer un nouveau fichier `{{tableobjet}}.py`` dans le dossier `/app/models`.  
Définir la classe en utilisant SQLAlchemy et en la faisant hériter de la classe `app.database.base_class.Base`.

Importer la classe dans le fichier `/app/models/__init__.py`. Cela permet d'avoir une meilleure syntaxe d'import dans les autres fichiers.

Importer la classe dans le fichier `/app/database/base.py`. L'objectif ici est que la classe soit disponible lors de l'import de la classe Base dans la configuration alembic.

2. Création du schéma Pydantic

Créer un nouveau fichier `{{tableobjet}}.py` dans le dossier `/app/schemas`. Ouvrir le fichier `/app/schemas/prediction_feedback.py` et copier son contenu dans le nouveau fichier créé. Changer les classes pour qu'elles correspondent aux données du nouvel objet.

Importer la classe dans le fichier `/app/schemas/__init__.py`. Cela permet d'avoir une meilleure syntaxe d'import dans les autres fichiers.

3. Création du CRUD

Ce fichier va contenir les fonctions permettant d'interagir avec BDD.

Créer un nouveau fichier `crud_{{tableobjet}}.py` dans le dossier `/app/crud`. Ouvrir le fichier `/app/crud/crud_prediction_feedback.py` et copier son contenu dans le nouveau fichier créé. Changer les types pour renseigner ceux précédemment créés.

Ne pas oublier de terminer le fichier par l'instanciation de la classe dans une variable.

Importer la variable dans le fichier `/app/crud/__init__.py`. Cela permet d'avoir une meilleure syntaxe d'import dans les autres fichiers.

4. Génération de la migration alembic

Ouvrir un nouveau terminal et saisir la commande suivante :
```shell
$ alembic revision --autogenerate -m "{{description of what you did}}"
```

La migration s'appliquera automatiquement lors du prochain redémarrage du dev container. Pour l'appliquer directement et, donc, mettre à jour la BDD, saisir la commande suivante :
```shell
$ alembic upgrade head
```

Si besoin, il est possible de revenir en arrière sur les migrations en utilisant la commande :
```shell
alembic downgrade {{identifiant_révision}}
```

Ou de revenir en arrière sur toutes les migrations via la commande :
```shell
alembic downgrade base
```