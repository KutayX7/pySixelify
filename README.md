# pySixelify
A relatively fast SIXEL converter utility, written purely in Python. [^1] [^2]
[^1]: It is quite fast if you consider that this is written entirely in Python, without hardware acceleration.
[^2]: Sixel, short for "six pixels", is a bitmap graphics format supported by terminals and printers from DEC. See https://en.wikipedia.org/wiki/Sixel for more details.

## Dependencies
* `pillow`: Python Imaging Library fork. [^3]
* A terminal that support SIXEL images (if you want to see the results).
  * VSCode's terminal (***tested***) [^4]
  * Windows Terminal (***tested***) [^5]
  * Konsole (***tested***) [^6]
  * XTerm
  * mlterm
  * WezTerm
  * Terminology
  * Exoterm
  * Gnuplot
  
[^3]: `pillow` is an optional dependency. You don't need it if you want to use this as a module and won't use any of the file input methods.

[^4]: VSCode `1.79+` supports Sixel images and all the features of pySixelify. If you can't see images, set `terminal.integrated.enableImage` to `true` (1.80+) or set `terminal.integrated.experimentalImageSupport` to `true` (1.79).
[^5]: The Terminal app for Windows supports Sixel images. Some configuration may be needed to use it. No register reuse.
[^6]: Konsole `22.04+` supports Sixel images. No configuration needed. No register reuse.

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
                                  which may not work with most terminals.
                                  It's experimental so please give feedback if you do use it <3
          -s, --silent : Ignores warning mesasges (if any).
         -p, --palette : Sets the palette generator algorithm.
                         Choices: QPUNM, OTFCD.
                         These only apply if there are more colors in the image
                         than there are color registers.
                         QPUNM is the default algorithm.
                         OTFCD is usually faster and sometimes can produce better results than QPUNM.
                         Try both!
```

## Example usage (as a command line tool)
* Read "test.png" and print it to the terminal
  * `python3 path/to/pySixelify.py "path/to/test.png"`
* Read "test.png" and save it to "output.sixel"
  * `python3 path/to/pySixelify.py "path/to/test.png" -o "path/to/output.sixel"`
* Read "test.png" and print it to the terminal, with only 16 color registers
  * `python3 path/to/pySixelify.py "path/to/test.png" -r 16`

## TO-DO
- [x] ~~Multiprocessing~~
- [ ] Global interpreter lock detection to switch between multi-threading and multi-processing
- [ ] Apply dithering when needed
- [ ] Automatic fallback to [libsixel](https://github.com/saitoha/libsixel) when it is possible and makes sense to (maybe?)
- [ ] Realtime SIXEL conversion for large, colorful images
- [ ] Ability to load and play videos
- [ ] Play Bad Apple on it in real-time, at minimum 30 FPS (must be done, one way or another)
- [ ] Lossless color output on ALL Sixel terminals
- [ ] ~~Remove the `pillow` dependency~~ (impractical, for now)

## Known issues
* Doesn't work on WASI (`concurrent.futures` library is not available)
* No multiprocessing on mobile platforms (`multiprocessing` library is not available)
* QPUNM loses information of low frequency (but important) colors
* OTFCD sometimes assigns wrong colors
* Pure Python is too slow for real-time conversion of large, colorful images
