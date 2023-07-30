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
2. Exécuter `scripts/docker-setup.sh` pour créer les conteneurs.
3. Lancer les conteneurs via Docker Desktop ou via ligne de commande.

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
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

VSCode doit faire automatiquement la redirection du port 8000.  
Ouvrir l’adresse ci-dessous dans un navigateur Web sur la machine hôte pour afficher la documentation :

http://localhost:8000/docs

### Monitoring

Le monitoring est mise en place avec la librairie [OpenTelemetry](https://opentelemetry.io) permettat l'envoi des évènements sur plusieurs solutions du marché. 
Dans ce projet, nous utilisons la version Open Source de [SigNoz](https://signoz.io).

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
__tablename__ = "prediction_feedback" # type: ignore
```

#### Docstring : Google

Nous utilisons la convention de [Google expliquée ici](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings).

L'extension autoDocstring, intégrée au dev container, permet de générer les templates de docstring via la commande (Cmd + Shift + p) `Generate Docstring`.

#### Linter : Pylint

Pylint analyse le contenu du fichier après la sauvegarde de celui-ci.

Parfois, Pylint fait des recommandations erronées. C'est le cas, par exemple, des validateurs de la librairie Pydantic :

```python
# Pylint(E0213:no-self-argument): Method 'assemble_db_connection' should have "self" as first argument
@validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: str | None, values: Dict[str, Any]) -> Any:
```

La désactivation des remarques de Pylint se fait via un commentaire formaté.   
Celui-ci commence par `pylint: disable=`, suivi par la(les) références de(s) remarque(s). 

- Pour désactiver une remarque sur une ligne seulement, on ajoute le commentaire sur la même ligne.
- Pour désactiver toutes les remarques d'un scope, on ajoute le commentaire sur une nouvelle ligne dans le scope. Attention, seule les lignes en dessous du commentaire seront ignorée.

```python
# Désactive la remarque E0213 seulement pour la ligne ci-dessous
@validator("SQLALCHEMY_DATABASE_URI", pre=True) # pylint: disable=E0213

# Désactive la remarque E0213 pour toutes les lignes dans le scope
# pylint: disable=E0213
@validator("SQLALCHEMY_DATABASE_URI", pre=True) 
```

#### Formateur : isort et Black

Formate automatiquement le code lors de l'enregistrement du fichier.

Attention, les imports sont réorganisés automatiquement. Dans certains cas, nous souhaitons maintenir un ordre précis.  
Pour désactiver la réorganisation des imports :
- Fichier complet : ajouter `# isort: skip_file` après la docstring du module.
- Bloc spécifique : encadrer les lignes avec les commentaires `# isort: off` et `# isort: on`.

[Documentation isort sur les méthode de désactivation](https://pycqa.github.io/isort/docs/configuration/action_comments.html)

### Test Unitaires fonctionnels sur l'API et modéle.

Ici est est question de faire différents tests unitaires sur le bon fonctionnement des différentes routes API mise en place et les bonnes informations sur la prédiction du modéle.
Dans le contexte de notre projet, la mise en place des tests unitaires constituent des bonnes pratiques de code dans le sens ou ils s'assurent que chaque élément constitutif du code soit en bonne santé. La majeure partie du code (l'insfrastructure) qui est le squelette du projet se trouvant dans le dossier Backend, nous devons tester certaines fonctionnalité. 
Ainsi nous devons savoir quels types de tests nous devons utilisés, soit un test unitaire ou test d'intégration.

##### Les bonnes pratiques pour effectuer des tests:

Les tests doivent se faire correctement à savoir:
 - Facile à écrire
 - Facile à lire et comprendre
 - Fiable...
Le code sera effectuer sur l'éditeur de programmation Visual Studio Code


