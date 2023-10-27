---
title: Documentation
draft: false
date: 2023-10-27
tags:
- dev
---
> [!caution] This page in under construction and might be incomplete.

 > [!todo]-
 > - [ ] Add a section regarding templates usage in Obsidian.
 > - [ ] Add a section to explain how to run the documentation locally.
## Introduction

This documentation is built using [Quartz v4.1.0.](https://quartz.jzhao.xyz/)
It allows us to use the directory `docs/content` as an [Obsidian vault](https://obsidian.md/) for a fast and visual editing allowing easy linking between pages.

Here is a comparison between how this page is displayed on Obsidian and how it will be rendered:

![[developer-documentation-obsidian_quartz_comparison.png]]
## Update the documentation
### Requirements

1. [Obsidian.md](https://obsidian.md/) installed for easy editing
2. [Git](https://git-scm.com/) to clone the repository, commit and push your modifications
3. [Github](https://github.com/) account

### Clone the repository and open the documentation in Obsidian

1. Clone the repository on your machine using this command in your terminal :
	```shell
	git clone git@github.com:JoffreyLGT/e-commerce-mlops.git
	```
2. Open Obsidian.
3. In Obsidian system menu, click on `File`, `Open vault`, `Open folder as vault`.
4. Use your system explorer to open the folder `e-commerce-mlops/docs/content`.

### Edit an existing page

Each note in Obsidian is transformed into a page. Therefore, you can search for the note (page) you want to edit within Obsidian and directly change it.

### Add a new page

Create a new note in Obsidian and paste the template bellow:

```md
---
title: Example Title
draft: false
date: YYYY-MM-DD
tags:
  - example-tag
---

The rest of your content lives here. You can use **Markdown** here :)
```

For more information regarding Authoring content, refer to [Quartz documentation](https://quartz.jzhao.xyz/authoring-content)
