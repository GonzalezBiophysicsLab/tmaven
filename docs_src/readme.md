# How to
This site is made using quarto.
The quarto source code is in `docs_src`, and this is rendered into `docs` which github pages will serve as the website.
To edit it, modify/write new .qmd files (it's a modified version of markdown.
To add files to the sidebar or update the website look, modify the _quarto.yml file
You can preview everything with 

```bash
quarto preview docs_src
```
## Publish changes.
* Make your changes. Use the quarto preview to make sure they're good.
* Render the site using `quarto render docs_src`.
* Use git to add and commit the changes (including the .html files etc that get generated in the `docs` folder
* github actions will update and deploy the site in about one minute
