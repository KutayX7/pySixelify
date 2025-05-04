# pySixelify
A relatively fast SIXEL converter utility, written purely in Python.
The only dependency is pillow, and that's optional.

## Dependencies
* `pillow`: Python Imaging Library fork. `pip install pillow`
  * **It's optional.** You only need it if you need to use this as a command line tool
  * or if you use the functions that need to load an image from a file.

## Comamnd line arguments
```
            [filename] : Name or path of the input file.
     -o, --output-file : Name or path of the output file.
                         if the `filename` is provided but not the `--output-file`,
                         the output will be printed to the terminal.
  -r, --register-count : The amount of color registers to use.
                         Can be: 1, 2, 4, 8, 16, 32, 64, 128, or 256 (default)
                         Warning: Setting it to 1 is a special case.
                         When set to 1, it will (re)use a single register to render EVERY color,
                         which may not work with every sixel terminal.
                         It's experimental so please give feedback if you use it <3
          -s, --silent : Ignores warning mesasges (if any).
```
