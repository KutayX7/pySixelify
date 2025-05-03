# pySixelify
A SIXEL converter utility written in Python.
The only dependency is pillow, and it's optional.

## Comamnd line arguments (requires pillow for now)
```
            [filename] : Name or path of the input file.
     -o, --output-file : Name or path of the output file.
                         if the `filename` is provided but not the `--output-file`,
                         the output will be printed to the terminal.
  -r, --register-count : The amount of color registers.
```
