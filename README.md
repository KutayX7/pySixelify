# pySixelify
A relatively fast SIXEL converter utility, written purely in Python. [^a] [^1]
[^a]: It is quite fast if you consider that this is written entirely in Python, without hardware acceleration. [^b]
[^1]: "Sixel, short for 'six pixels', is a bitmap graphics format supported by terminals and printers from DEC." See https://en.wikipedia.org/wiki/Sixel for more details.
[^b]: This projects focuses on converting images (that are already decoded in the memory) into the sixel format, not on loading images from files. The way `pillow` reads those files is not accounted for.

## Requirements
* `pillow`: Python Imaging Library (fork). [^c]
* A terminal that support SIXEL images (if you want to see the results, which you probably do).
  * VSCode's terminal (***tested***) [^d]
  * Windows Terminal (***tested***) [^e]
  * Konsole (***tested***) [^f]
  * XTerm
  * mlterm
  * WezTerm
  * Terminology
  * Exoterm
  * Gnuplot

[^c]: `pillow` is optional. You don't need it if you want to use this as a module and won't use any of the file input methods.

[^d]: VSCode `1.79+` supports Sixel images and all the features of pySixelify. If you can't see images, set `terminal.integrated.enableImage` to `true` (1.80+) or set `terminal.integrated.experimentalImageSupport` to `true` (1.79).
[^e]: The Terminal app for Windows supports Sixel images. Some configuration may be needed to use it. No register reuse.
[^f]: Konsole `22.04+` supports Sixel images. No configuration needed. No register reuse.

## Comamnd line arguments
```
  filename              Name or path of the input image file.
                        All the other arguments are optional.

  -s, --silent          Ignores some warning mesasges (if any).

  -o, --output-file <path>      Name or path of the output file.
                                If the `filename` is provided but not the `--output-file`,
                                the result will be written to the standard output.
                                DO NOT USE PIPES to save the output!

  -r, --register-count <value>  The amount of color registers to use.
                                Choices: 1, 2, 4, 8, 16, 32, 64, 128, or 256 (default).
                                WARNING: Setting it to 1 is a special case.
                                  When set to 1, it will (re)use a single color register to
                                  render EVERY color, which won't work on most terminals.
                                  It's experimental so please give feedback if you do use it <3

  -p, --palette        <value>  Sets the palette generator algorithm.
                                Choices: QPUNM, OTFCD.
                                These only apply if there are more colors in the image
                                than there are color registers.
                                QPUNM is the default algorithm, for now.
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

> [!WARNING]
> When saving to a file, please use the `-o <path>` argument.<br>
> Pipes may work as well, for now, but that might change in the future.

## API (not stable)
> [!WARNING]
> Do not rely on any method, or variable, that starts with an underscore as they are supposed to be private and can break at any moment without a warning.

> [!TIP]
> Never set the `palette_generation_algorithm` unless you really have to or just messing around. The default values should be better. This is likely to be more important later.

```Python
print_image_from_path(path: str, *, register_count: int = 256, palette_generation_algorithm: PaletteGenerationAlgorithm = DEFAULT_PALETTE_GENERATION_ALGORITHM)

from_file_to_file(input_path: str, output_path: str, *, register_count: int = 256, palette_generation_algorithm: PaletteGenerationAlgorithm = DEFAULT_PALETTE_GENERATION_ALGORITHM)

img2sixels(image: List[List[Tuple[int, int, int, int]]], *, register_count: int = 256, palette_generation_algorithm: PaletteGenerationAlgorithm = DEFAULT_PALETTE_GENERATION_ALGORITHM) -> str
```

## TO-DO
- [x] ~~Multiprocessing~~
- [ ] Global interpreter lock detection to switch between multi-threading and multi-processing
- [ ] Optional dithering
- [ ] Realtime SIXEL conversion for large images
- [ ] Ability to load and play videos
- [ ] Auto select the optimal palette generator based on the image or video, if not explicitly specified
- [ ] Play Bad Apple on it in real-time, at minimum 30 FPS (must be done, one way or another)
- [ ] Lossless true color output on all supported terminals
- [ ] Remove all third-party dependencies

## Known issues
* Doesn't work on WASI (`concurrent.futures` library is not available)
* No multiprocessing on mobile platforms (`multiprocessing` library is not available)
* QPUNM is usually good enough but better options are definitely needed
* Quality of OTFCD seems to be entirely dependent on a number that is too sensitive and the current way we calculate that number is just a very rough approximation
* Pure Python is too slow for real-time conversion of large detailed images
