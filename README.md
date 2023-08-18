# Catégorisation des produits d'un boutique en ligne

Fourni une API permettant de prédire la catégorie d'un produit en fonction de sa désignation, description et d'une image.

## Structure du projet

```
E-COMMERCE-MLOPS/  
├─ .github/workflow/ : Github Actions  
├─ .vscode/settings.json : configurations du workspace VSCode  
├─ backend/ : API et modèle de prédiction  
├─ frontend/ : (plus tard) application WEB pour interagir avec l'API  
├─ scripts/ : liste des scripts permettant de lancer les tests ou de faire le déploiement 
│ ├─ ressources/ : contient les ressources nécessaires aux scripts
```

## Mise en place avec Docker

1. Remplir le fichier .env avec vos informations.
2. Ajouter la variable d'environnement `export ENV_TARGET="development"`
3. Exécuter `scripts/docker-deploy.sh` pour créer les conteneurs.
4. Lancer les conteneurs via Docker Desktop ou via ligne de commande.

## Backend

### Configuration de l'environnement de développement

Un conteneur de développement a été préparé.  
Voici les prérequis : 
- [Visual Studio Code](https://code.visualstudio.com)
- Extension VSCode [Dev Container](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

Une fois les extensions installées, veuillez-suivre les étapes suivantes :  
1. Cloner le repo :
```shell
git clone https://github.com/JoffreyLGT/e-commerce-mlops.git
```
2. Ouvrir le projet dans VSCode.
3. Ouvrir la palette des commandes (`Cmd+Shift+p`).
4. Saisir **dev container open** et sélectionner l'option **Dev Containers: Open folder in Container...**.
5. Sélectionner le dossier **e-commerce-mlops**.

La fenêtre de VSCode va se recharger. Une fois l'installation du dev container terminée, plusieurs terminaux vont s'ouvrir :
- backend : terminal se trouvant dans le dossier backend avec l'environnement virtuel du projet backend.
- datascience : terminal se trouvant dans le dossier backend avec l'environnement virtuel du projet datascience.
- API reload : lance l'API en mode rechargement.
- MLFlow UI : lance l'interface Web de MLFlow.

Les extensions peuvent afficher des notifications lors de la configuration, notamment Pylance indiquant que l'extension Python n'est pas détectée. Il faut simplement les fermer sans les prendre en compte.

### Lancement de l'API

Le lancement de l'API en mode développement sur le conteneur se fait avec le script `start-reload.sh` :

```shell
./scripts/start-reload.sh
```

VSCode s'occupe automatiquement de la redirection du port 8000.  
Ouvrir l’adresse ci-dessous dans un navigateur Web sur la machine hôte pour afficher la documentation :

http://localhost:8000/docs

### Monitoring

Le monitoring est mise en place avec la librairie [OpenTelemetry](https://opentelemetry.io) permettant l'envoi des évènements sur plusieurs solutions du marché. 
Dans ce projet, nous utilisons la version Open Source de [SigNoz](https://signoz.io).

Pour démarrer l'application avec la télémétrie, il faut exécuter le script `start-with-telemetry.sh` : 

```shell
./scripts/start-with-telemetry.sh
```

A noter que Signoz doit être installé sur votre machine et connecté sur le même réseau que le devcontainer.

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

## Normes de développement

### Git

#### Pull Request
Les PR doivent obligatoirement être liées à une issue et avoir une description permettant d'aider le validateur dans ses tests.

#### Messages de commit

Doivent être rédigés en anglais et respecter le format ci-dessous :

```
Short (72 chars or less) summary

More detailed explanatory text. Wrap it to 72 characters. The blank
line separating the summary from the body is critical (unless you omit
the body entirely).

Write your commit message in the imperative: "Fix bug" and not "Fixed
bug" or "Fixes bug." This convention matches up with commit messages
generated by commands like git merge and git revert.

Further paragraphs come after blank lines.

- Bullet points are okay, too.
- Typically a hyphen or asterisk is used for the bullet, followed by a
  single space. Use a hanging indent.
```

Exemple de message de commit :

```
Add CPU arch filter scheduler support

In a mixed environment of…
```

### Python

Le dev container est configuré pour installer et configurer automatiquement les extensions VSCode. 

#### Commentaires

Doivent être rédigés en anglais.

#### Language server : Pylance

Parfois, Pylance indique des erreurs de type à cause de décorateurs des librairies. C'est le cas notamment de la propriété `__tablename__` de SQLAchemy.

Pour ignorer ces erreurs, ajouter le commentaire suivant en bout de ligne :
```python
__tablename__ = "prediction_feedback" # pyright: ignore
```

#### Docstring : Google

Nous utilisons la convention de [Google expliquée ici](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings).

L'extension autoDocstring, intégrée au dev container, permet de générer les templates de docstring via la commande (Cmd + Shift + p) `Generate Docstring`.

#### Analyseur de code statique : Ruff

[Ruff](https://beta.ruff.rs/docs/) effectue l'analyse de code statique du fichier en direct et fait des recommandations. Il combine des règles provenant de plusieurs autres linter du marché et réorganise les imports en utillisant les règles [iSort](https://beta.ruff.rs/docs/faq/#how-does-ruffs-import-sorting-compare-to-isort).

Pour visualiser les règles activées dans le projet, ouvrir le fichier `/backend/pyproject.toml`.

Certaines recommandations peuvent être érronées. Pour les désactiver, je vous invite à consulter la page  [Ruff error suppression](https://beta.ruff.rs/docs/configuration/#error-suppression).

#### Formateur : Black

[Black](https://black.readthedocs.io/en/stable/) formate automatiquement le code lors de l'enregistrement du fichier.

#### Vérificateur de type : Mypy

[Mypy](https://mypy.readthedocs.io/en/stable/) s'occupe de mettre en avant les potentiels problèmes liés aux types.

Il s'exécute à l'enregistrement du fichier.
