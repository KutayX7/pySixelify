# pySixelify
A SIXEL converter utility written in Python.
The only dependency is pillow, and it's optional.

## Comamnd line arguments (requires pillow, for now)
```
            [filename] : Name or path of the input file.
     -o, --output-file : Name or path of the output file.
                         if the `filename` is provided but not the `--output-file`,
                         the output will be printed to the terminal.
  -r, --register-count : The amount of color registers to use.
                         Can be: 1, 2, 4, 8, 16, 32, 64, 128, 256
                         Warning: Setting it to 1 is a special case.
                         When set to 1, it will (re)use a single register to render EVERY color,
                         which may not work with every sixel terminal.
                         It's experimental so please give feedback if you use it <3
          -s, --silent : Ignores warning mesasges (if any).
```
