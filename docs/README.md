# Generating documentation

Documentation can be generated with:
```
sphinx-apidoc -o ./source ../discretesampling
make clean
make html
```

Overwriting existing `.rst` files can be enabled by adding the `-f` flag to the
call to `sphinx-apidoc`:
```
sphinx-apidoc -f -o ./source ../discretesampling
```
