# How to update
In this tMANUAL folder:

``` bash
quarto render
```

This will fail if you don't have `xelatex` installed. If you use tectonic (`brew install tectonic`), you can then:
``` bash
mv index.tex tMANUAL.tex
tectonic tMANUAL.tex
rm tMANUAL.tex
```

# Notes
You MUST use `index.qmd` as a file in your contents. See <https://github.com/quarto-dev/quarto-cli/discussions/3759>.
The web version of tMANUAL uses `web_intro.qmd` as the first page instead