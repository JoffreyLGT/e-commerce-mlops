# Product classification

## Easy setup
- Open VSCode
- Press F1, select `Dev Containers: Open Workspace in Container...`
- Select `e-commerce-mlops.code-workspace`

## Sub-projects information

Each sub-project has their own tasks to get them up and running.  
To execute a task, press F1 and choose `Tasks: Run Task`.

### Root

Files bellow are references in sub-projects through symlinks:
| File      | Sub-projects                  |
| --        | --                            | 
| mypy.ini  | backend, datascience          |
| **.env**  | root, backend, datascience    |
