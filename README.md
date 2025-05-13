# pySixelify
A relatively fast SIXEL converter utility, written purely in Python.

## Dependencies
* `pillow`: Python Imaging Library fork. `pip install pillow`
  * **It's optional.** You only need it if you need to use this as a command line tool
  * or if you use the functions that need to load an image from a file.
* A terminal that support SIXEL images.
  * (**It's also optional.** Not needed if you don't need to see the results.)
  * VSCode's terminal (***tested***, fully functional, may need some configuration)
  * Windows 11 Terminal (***tested***, mostly functional, no register reuse, may need some configuration)
  * XTerm
  * mlterm
  * WezTerm
  * Konsole
  * Terminology
  * Exoterm
  * Gnuplot
  

## Comamnd line arguments
```
            [filename] : Name or path of the input file.
     -o, --output-file : Name or path of the output file.
                         If the `filename` is provided but not the `--output-file`,
                         the result will be written to the standard output.
  -r, --register-count : The amount of color registers to use.
                         Choices: 1, 2, 4, 8, 16, 32, 64, 128, or 256 (default).
                         WARNING: Setting it to 1 is a special case.
                                  When set to 1, it will (re)use a single register to render EVERY color,
                                  which may not work with every sixel terminal.
                                  It's experimental so please give feedback if you do use it <3
          -s, --silent : Ignores warning mesasges (if any).
         -p, --palette : Sets the palette generator algorithm.
                         Choices: QPUNM, OTFCD.
                         QPUNM is the default algorithm, usually high quality
                         OTFCD is the new, blazingly fast algorithm; but provides lower quality
```

## Example usage (as a command line tool)
* Read "test.png" and print it to the terminal
  * `python pySixelify "test.png"`
* Read "test.png" and save it to "test.sixel"
  * `python pySixelify "test.png" -o "test.sixel"`
* Read "test.png" and print it to the terminal, with only 16 colors
  * `python pySixelify "test.png" -r 16`

## TO-DO
* More efficient palette generators
* Remove the `pillow` dependency
* Realtime SIXEL conversion
* Video player
* ...
* Play Bad Apple on it in real-time at minimum 60 FPS
