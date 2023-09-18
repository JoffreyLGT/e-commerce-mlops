<!-- markdownlint-disable MD013 -->

# Catégorisation des produits d'un boutique en ligne

![Python version](https://img.shields.io/badge/Python-3.11-blue.svg)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![mypy](https://img.shields.io/badge/checked-mypy?label=mypy)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![VS Code Container](https://img.shields.io/static/v1?label=VS+Code&message=Container&logo=visualstudiocode&color=007ACC&logoColor=007ACC&labelColor=2C2C32)](https://open.vscode.dev/JoffreyLGT/e-commerce-mlops)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)

API permettant de prédire la catégorie d'un produit en fonction de sa désignation, description et d'une image.

![Déploiement](https://gist.githubusercontent.com/JoffreyLGT/b0b0fed9dd9b77c46727a67c2eed583a/raw/84df86a25031d6c8d7f366329fe62561ee21d56a/docker-deploy.svg)

## Structure du projet

```text
E-COMMERCE-MLOPS/
├─ .devcontainer/ : fichiers du conteneur de dev
├─ .github/workflow/ : Github Actions
├─ .vscode/tasks.json : tâches du project global
├─ backend/ : API de prédiction (sous-projet)
├─ datascience/ : MLOPS via MLFlow (sous-projet)
├─ frontend/ : placeholder pour une éventuelle application WEB pour interagir avec l'API
├─ scripts/ : liste des scripts globaux du projet
│ ├─ ressources/ : contient les ressources nécessaires aux scripts
│ ├─ tests/ : sous-scripts utilisés par `run-tests.sh`
├─ .env : définition des variables d'environnement
```

## Mise en place avec Docker

1. Remplir le fichier [`.env`](.env) avec les informations du projet.
2. Toujours dans [`.env`](.env), vérifier que la valeur de la variable `ENV_TARGET` est renseignée.
3. Exécuter [`scripts/docker-deploy.sh`](scripts/docker-deploy.sh) pour créer les conteneurs.
4. Lancer les conteneurs via Docker Desktop ou via ligne de commande :

```shell
docker compose -f docker-compose.yaml up -d
```

## Configuration de l'environnement de développement

### Local

Utilisation de la machine locale pour le développement permettant l'utilisation du **GPU pour les tâches Tensorflow**.
*Actuellement configuré pour les utilisateurs de MacOS.*

Prérequis :

- Système UNIX-like (Linux, MacOS). Les utilisateurs de Windows peuvent soit utiliser le [conteneur de développement](#conteneur-de-développement) mis à disposition, soit installer un système Linux sur [WSL2](https://learn.microsoft.com/fr-fr/windows/wsl/install).
- [Visual Studio Code](https://code.visualstudio.com)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Python 3.11](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installing-with-pipx) (installation via [pipx](https://pypa.github.io/pipx/) recommandée)
- [Pre-commit](https://pre-commit.com) pour la gestion des hook Git.

Mise en place :

1. Cloner le repo :

    ```shell
    git clone https://github.com/JoffreyLGT/e-commerce-mlops.git
    ```

2. Ouvrir VSCode
3. Cliquer sur `File`, `Open Workspace from File...`
4. Sélectionner le fichier [e-commerce-mlops.code-workspace](e-commerce-mlops.code-workspace)
5. Ouvrir le fichier [.env](.env) et vérifier que les variables d'environnement ci-dessous sont configurées avec ces valeurs :

    ```env
    USE_DB_CONTAINER=true
    POSTGRES_SERVER=localhost
    ```

6. Une fois le projet ouvert, ouvrir un terminal dans le dossier `root`, et saisir la commande ci-dessous :

    ```shell
    ./scripts/environment-setup.sh
    ```

7. Installer les [extensions recommandées dans le workspace.](#extensions)

### Conteneur de développement

Utilisation d'un conteneur Docker possédant tous les outils de développement.

Voici les prérequis :

- [Visual Studio Code](https://code.visualstudio.com)
- Extension VSCode [Dev Container](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

Une fois les prérequis installées, veuillez-suivre les étapes suivantes :

1. Cloner le repo :

    ```shell
    git clone https://github.com/JoffreyLGT/e-commerce-mlops.git
    ```

2. Ouvrir le projet dans VSCode.
3. Ouvrir la palette des commandes (`Cmd+Shift+p`).
4. Saisir `dev open workspace` et sélectionner l'option `Dev Containers: Open Workspace in Container...`.
5. Sélectionner le fichier `e-commerce-mlops.code-workspace`.

La fenêtre de VSCode va se recharger et le conteneur de développement va se mettre en place.

Les [extensions recommandées dans le workspace](#extensions) sont installées automatiquement. Cependant, VSCode peut afficher des notifications lors de leur installation, notamment Pylance indiquant que l'extension Python n'est pas détectée. Il faut simplement les fermer sans les prendre en compte.

### Extensions

Le workspace contient une liste d'extensions recommandées. Pour les installer, ouvrir la Command Palette (`CMD + Shift + p` ou via le menu `View > Command Palette...`) et rechercher `Extensions: Show Recommended Extensions`.

Sur la ligne `WORKSPACE RECOMMENDATION`, cliquer sur `Install workspace Recommended Extensions`.

## Questions et réponses

Cette section concerne les outils en général. Chaque sous-projet a sa propre section :

- [root](#questions-et-réponses-(root))
- [backend](#questions-et-réponses-(backend))
- [datascience](#questions-et-réponses-(datascience))

### Docker m'indique que je le disque de la VM est plein

Docker stocke les images, containers et volumes dans un disque virtuel. Celui-ci a une taille définie dans les paramêtres de Docker Desktop.

Pour solutionner cette erreur, deux solutions :

1. Augmenter la taille du disque via Docker Desktop, rubrique `Settings`.
2. Vider tout le contenu du disque et repartir de zéro avec la commande ci-dessous :

```shell
docker system prune -a --volumes
```

### Mon ordinateur surchauffe pendant le développement / utilisation intensive du CPU

- Ouvrir le moniteur d'activité pour repérer le processus utilisant le CPU.
- S'il s'agit de `code helper (plugin)`, noter le PID.
- Quitter complètement VSCode (`Cmd + Q` sur MacOS)

    Si le processus disparait, relancer VSCode et attendre quelques minutes. Si le processus ne revient pas, tout est de nouveau en ordre. Sinon, continuer les étapes ci-dessous.

- Ouvrir un terminal et saisir la commande suivante en replaçant 67377 par le PID noté précédemment :

    ```shell
    ps aux | grep 67377
    ```

- Le résultat de la commande à le format suivant :

    ```text
    joffrey 67377 110.4  8.1 1587151776 1351248   ??  R     4:35PM  29:40.37 [...] /Users/joffrey/.vscode/extensions/ms-python.vscode-pylance-2023.8.30/dist [...]
    ```

    *Note : [...] indique que le contenu a été tronqué.*

    En analysant la ligne, nous constatons la mention `/.vscode/extensions/ms-python.vscode-pylance-2023.8.30/` : [Pylance](#language-server--pylance) est donc le coupable.

### Comment renseigner le schéma d'un fichier TOML pour bénéficier de l'auto-completion ?

- Ouvrir le site [Schemastore.org](https://www.schemastore.org/json/) et rechercher le schéma correspondant à votre fichier.
- Ouvrir votre fichier TOML.
- Insérer la valeur ci-dessous en première ligne du fichier :

    ```toml
    #:schema https://json.schemastore.org/ruff.json

    # Add your values after
    ```

### Comment tester le projet sur d'autres architectures

Nous utilisons principalement des machines ARM (Apple Silicon) pour le développement du projet.
Afin de pouvoir tester sur des machines de type AMD64, nous utilisons le conteneur de développement.

Pour cela, il faut :

- Ouvrir le fichier [.devcontainer/Dockerfile](.devcontainer/Dockerfile)
- Modifier la première ligne pour y ajouter `--platform=linux/amd64` comme ceci :

    ```dockerfile
    FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/python:1-3.11-bullseye
    ```

## Information sur les projets

Un workspace VSCode est mis à disposition à la racine du projet sous le nom *[e-commerce-mlops.code-workspace](e-commerce-mlops.code-workspace)*. **Il est fortement recommandé de l'utiliser.**

Il s'agit d'un multi-root workspace configuré pour permettre un fonctionnement optimal des sous-projets. Chaque sous-projet contient des tâches VSCode permettant d'exécuter les fonctions principales.

### root (racine du projet)

Contient les fichiers commun à tout le projet, les sous-projets ainsi que des fichiers utilisés dans les sous-projets.

#### Symlinks

Attention, certains fichiers sont référencés comme symlinks dans les sous-projets:
| File      | Sub-projects                  |
| --        | --                            |
| .env      | backend, datascience          |
| mypy.ini  | backend, datascience          |
| ruff.toml | backend, datascience          |

#### VSCode Tasks (root)

Les tâches ci-dessous sont disponibles :
| Nom       | Description                |
| --        | --                            |
| Run all Git hooks  | Permet de lancer tous les hook Git sans faire d'action Git.          |
| Run Git pre-commit hooks  | Permet de lancer les hook git de pre-commit sans avoir à faire de commit.          |
| Run Git pre-push hooks  | Permet de lancer les hook Git de pre-push sans avoir à faire de commit.          |

#### Questions et réponses (root)

Rien pour le moment.

### backend

Permet de lancer l'API de catégorisation des produits.

#### Démarrage en mode développement

Le lancement de l'API en mode développement sur le conteneur se fait avec le script `start-reload.sh` :

```shell
./scripts/start-reload.sh
```

VSCode s'occupe automatiquement de la redirection du port 8000.
Ouvrir l’adresse ci-dessous dans un navigateur Web sur la machine hôte pour afficher la documentation :

<http://localhost:8000/docs>

Une tâche portant le nom `Start API in reload mode` permet de lancer l'API en mode développement.

#### Monitoring et démarrage en mode staging / production

Le monitoring est mise en place avec la librairie [OpenTelemetry](https://opentelemetry.io) permettant l'envoi des évènements sur plusieurs solutions du marché.
Dans ce projet, nous utilisons la version Open Source de [SigNoz](https://signoz.io).

Pour démarrer l'application avec la télémétrie, il faut exécuter le script `start-with-telemetry.sh` :

```shell
./scripts/start-with-telemetry.sh
```

A noter que Signoz doit être installé sur votre machine et connecté sur le même réseau que le devcontainer.

#### Tests automatisés

Les tests sont exécutés par [Pytest](https://docs.pytest.org/en/7.4.x/). Ils se lancent soit via la fonction [Testing de VSCode](https://code.visualstudio.com/docs/python/testing#_run-tests), soit en exécutant la commande suivante :

```shell
poetry run ./scripts/start-tests.sh
```

#### Git hook

Les hook sont générés automatiquement dans le dossier [.git/hooks](.git/hooks/) par l'utilitaire [pre-commit](https://pre-commit.com) lors de l'exécution de [scripts/environment-setup.sh](scripts/environment-setup.sh).
Ceux-ci ont été sélectionnés pour être rapides et le moins intrusifs possible pour ne pas géner le workflow des développeurs.

Le projet en utilise trois types :

- pre-commit : principalement le formatage des fichiers et des vérifications statiques.
- pre-push : analyse de code statique.
- commit-msg : vérifie le message de commit pour s'assurer que celui-ci correspond à nos [standards.](#messages-de-commit)

#### VSCode Tasks (backend)

Les tâches ci-dessous sont disponibles :
| Nom       | Description                |
| --        | --                            |
| Start API in reload mode | Lance l'API en mode reload, permettant de recharger automatiquement le code lorsqu'il est modifié. |
| Check Tensorflow GPU support | Affiche un message pour indiquer si Tensorflow supporte des GPU dans l'environnement. |
| Start PostgreSQL DB container | Démarre le conteneur *db* depuis le fichier [docker-compose.yaml](docker-compose.yaml). |
| Stop PostgreSQL DB container | Stop le conteneur *db* démarré depuis la tâche Start PostgreSQL DB container. |

#### Questions et réponses (backend)

##### Comment ajouter ou modifier une table en BDD ?

>> TLDR : Ajouter ou modifier les modèles SQLAlchemy dans `/app/models/`, les schemas Pydantic dans `/app/schemas/` et les outils CRUD d'interaction avec la BDD dans `/app/crud/`. Attention à bien mettre à jour les fichier `__init__.py` de ces modules ! Pour terminer, générer la migration alembic.
>> Utiliser **prediction_feedback** comme exemple.

1. Création du modèle SQLAlchemy

    Créer un nouveau fichier `{{tableobjet}}.py` dans le dossier`/app/models`.
    Définir la classe en utilisant SQLAlchemy et en la faisant hériter de la classe`app.database.base_class.Base`.

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
    alembic revision --autogenerate -m "{{description of what you did}}"
    ```

    La migration s'appliquera automatiquement lors du prochain redémarrage du dev container. Pour l'appliquer directement et, donc, mettre à jour la BDD, saisir la commande suivante :

    ```shell
    alembic upgrade head
    ```

    Si besoin, il est possible de revenir en arrière sur les migrations en utilisant la commande :

    ```shell
    alembic downgrade {{identifiant_révision}}
    ```

    Ou de revenir en arrière sur toutes les migrations via la commande :

    ```shell
    alembic downgrade base
    ```

### datascience

#### Création d'un jeu de données

En partant du principe que toutes les données ont été stockées dans le dossier `data/datasets/original` avec la structure suivante :

```text
original
├── X.csv
├── y.csv
└──── images
   ├── image_977803476_product_278535420.jpg
   ├── image_1174586892_product_2940638801.jpg
   └── [...]
```

- `X.csv` contient les features, à savoir les informations sur le produit dans les colonnes `désignation`, `description`, `productid` et `imageid`.
- `y.csv` contient la target, à savoir la catégorie du produit dans une colonne.
- `images` contient toutes les images des produits. Les images doivent respecter la convention `image_[imageid]_[productid].jpg`

Pour créer un dataset (jeu de données) pour l'entrainement, utiliser le script [optimize_images.py](datascience/scripts/optimize_images.py).

```bash
# Depuis le dossier e-commerce-mlops/datascience
python -m scripts.optimize_images --train-size 0.8 --test-size 0.2 --input-dir "data/originals" --output-dir "data/datasets/example"
```

Le script va s'occuper de traiter les images pour retirer les bandes blanches inutiles, les redimentionner et va créer les jeux de données d'entrainement (80%) et de test (20%) au format `.csv`.

#### Entrainement d'un modèle

Le projet utilise 3 modèles (image, texte et fusion). De ce fait, 3 scripts sont mis à disposition pour l'entrainement :

- [train_image_model.py](datascience/train_image_model.py)
- [train_text_model.py](datascience/train_text_model.py)
- [train_fusion_model.py](datascience/train_fusion_model.py)

Ceux-ci utilisent [MLFlow](https://mlflow.org) pour le suivi des metrics.

Chaque script est documenté. Pour afficher l'aide, utiliser la commande `--help`.  
Exemple :

```bash
# Affiche l'aide du script train_image_model.py
python -m scripts.train_image_model --help
```

Ce qui affiche :

```text
usage: train_image_model.py [-h] [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR] [--batch-size BATCH_SIZE] [--no-data-augmentation]
                            [--train-patience TRAIN_PATIENCE]

Create and train a new image model using dataset provided with --input-dir, then save it to --output-dir with its performance
statistics.

required arguments:
  --input-dir INPUT_DIR
                        Directory containing the datasets to use.
  --output-dir OUTPUT_DIR
                        Directory to save trained model and stats.

optional arguments:
  --batch-size BATCH_SIZE
                        Size of the batches to use for the training. Set as much as your machine allows. (default: 96)
  --no-data-augmentation
                        Add layers of data augmentation to avoid early overfitting. (default: True)
  --train-patience TRAIN_PATIENCE
                        Number of epoch to do without improvements before stopping the model training. (default: 10)

help:
  -h, --help            show this help message and exit
```

#### Mise en ligne d'une nouvelle version du modèle

1. Dans un terminal, lancer l'interface graphique de MLFlow avec la commande suivante :

    ```bash
    mlflow ui
    ```

2. Dans l'en-tête, cliquer sur `Experiments`.
3. Rechercher le run du modèle à mettre en production et cliquer dessus.
4. Dans la liste des `artifacts`, cliquer sur le modèle, puis sur `Register Model`.
5. Dans la popup qui s'ouvre, sélectionner le modèle en question, puis cliquer sur `Register`.
6. Dans l'en-tête, cliquer sur le modèle, puis sur la version.
7. Le champ `Stage` permet de changer le statut du modèle (`Staging` pour test, `Production` pour production.
8. Utiliser la commande ci-dessous pour lancer le modèle :

    ```bash
    # Remplacer image par le nom du modèle, et Staging par Production pour une mise en production
    mlflow models serve -m "models:/image/Staging" --port 5002 --env-manager local
    ```

#### Envoyer une requête sur le modèl

L'exemple ci-dessous effectue une prédiction sur le modèle image :

```bash
curl -d '{"dataframe_records":[{"image path":"/path/to/image_1261427590_product_3898729018.jpg"}]}' -H 'Content-Type: application/json' -X POST localhost:5002/invocations
```

Le résultat est le suivant (l'utilisation d'un formater JSON permet une meilleure lisibilité) :

```json
{"predictions": [[{"category_id": 1301, "probabilities": 17.82}, {"category_id": 2220, "probabilities": 16.83}, {"category_id": 1280, "probabilities": 16.58}, {"category_id": 1560, "probabilities": 9.92}, {"category_id": 1320, "probabilities": 9.25}, {"category_id": 2060, "probabilities": 6.27}, {"category_id": 2582, "probabilities": 2.85}, {"category_id": 1281, "probabilities": 2.7}, {"category_id": 1302, "probabilities": 2.14}, {"category_id": 2522, "probabilities": 1.98}, {"category_id": 1140, "probabilities": 1.78}, {"category_id": 1180, "probabilities": 1.6}, {"category_id": 60, "probabilities": 1.33}, {"category_id": 2585, "probabilities": 1.21}, {"category_id": 10, "probabilities": 1.18}, {"category_id": 2403, "probabilities": 1.18}, {"category_id": 2280, "probabilities": 1.08}, {"category_id": 2705, "probabilities": 0.96}, {"category_id": 50, "probabilities": 0.89}, {"category_id": 1300, "probabilities": 0.5}, {"category_id": 1940, "probabilities": 0.47}, {"category_id": 40, "probabilities": 0.45}, {"category_id": 2583, "probabilities": 0.38}, {"category_id": 2905, "probabilities": 0.33}, {"category_id": 2462, "probabilities": 0.32}, {"category_id": 1920, "probabilities": 0.01}, {"category_id": 1160, "probabilities": 0.0}]]}%
```

#### Questions et réponses (datascience)

Rien pour le moment

## Normes de développement

### Git

#### Pull Request

Les PR doivent obligatoirement être liées à une issue et avoir une description permettant d'aider le validateur dans ses tests.

#### Messages de commit

Nous utilisons la convention [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

Une extension VSCode est recommandée dans le workspace permettant de rédiger les messages façilement. Pour l'utiliser :

1. Ouvrir la palette des commandes (`Cmd+Shift+p`).
2. Cliquer sur `Conventional Commits`.
3. Suivre les indication.

Le message de commit sera alors automatiquement ajouté dans la vue [`Source control`](https://code.visualstudio.com/docs/sourcecontrol/overview).

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

Les configurations, incluant les règles activées, se trouvent dans le fichier des configuration Ruff : [`ruff.toml`](ruff.toml).

Certaines recommandations peuvent être érronées. Pour les désactiver, consulter la page  [Ruff error suppression](https://beta.ruff.rs/docs/configuration/#error-suppression).

Il est recommandé d'ajouter un message pour expliquer pourquoi la règle est désactivée.

#### Formateur Python: Black

[Black](https://black.readthedocs.io/en/stable/) formate automatiquement le code lors de l'enregistrement du fichier.

Les configurations se trouvent dans les fichier `pyproject.toml` de chaque sous-projet.

#### Vérificateur de type : Mypy

[Mypy](https://mypy.readthedocs.io/en/stable/) s'occupe de mettre en avant les potentiels problèmes liés aux types.

Il s'exécute à l'enregistrement du fichier.

Les configurations se trouvent dans le fichier [`mypy.ini`](mypy.ini).
