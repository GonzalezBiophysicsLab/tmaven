# How to use pup

might have had issues with underscores in the module name. Removed them for now...
actually they were fine.. i put them back in and it works fine.

``` bash
python -m venv env
source env/bin/activate
pip install pup
pip install ./
pup package ./ --icon-path ./tmaven/interface/resources/logo.png --nice-name tMAVEN --license-path ./LICENSE   
```
