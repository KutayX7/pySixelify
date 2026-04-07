# pySixelify
A relatively fast SIXEL converter utility, written purely in Python. [^a] [^1]
[^a]: It is quite fast if you consider that this is written entirely in Python, without hardware acceleration. [^b]
[^1]: "Sixel, short for 'six pixels', is a bitmap graphics format supported by terminals and printers from DEC." See https://en.wikipedia.org/wiki/Sixel for more details.
[^b]: This projects focuses on converting images (that are already decoded in the memory) into the sixel format, not on loading images from files. (The way `pillow` reads and decodes those files is not accounted for.)

## Requirements
* Python 3.7+
* `pillow`: Python Imaging Library (fork) [^c]
* A terminal that support SIXEL images (if you want to see the results, which you probably do)
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

[^d]: VSCode (1.79+) supports Sixel images and all the features of pySixelify. If you can't see images, set `terminal.integrated.enableImage` to `true` (1.80+) or set `terminal.integrated.experimentalImageSupport` to `true` (1.79).
[^e]: The Terminal app for Windows supports Sixel images. Some configuration may be needed to use it. No register reuse.
[^f]: Konsole `22.04+` supports Sixel images. No configuration needed. No register reuse.

## Command line arguments
```
  filename              Name or path of the input image file.
                        All the other arguments are optional.

  -s, --silent          Ignores some warning mesasges (if any).

  -o, --output-file <path>      Name or path of the output file.
                                If the `filename` is provided but not the `--output-file`,
                                the result will be written to the standard output.
                                DO NOT USE PIPES to save the output!

  -r, --register-count <value>  The amount of color registers to use.
                                Choices: 1, 2, 4, 8, 16, 32, 64, 128, or 256 (default)
                                WARNING: Setting it to 1 is a special case.
                                    When set to 1, it will (re)use a single color register to
                                    render EVERY color, which won't work on most terminals.
                                    It's experimental so please give feedback if you do use it! <3

  -p, --palette        <value>  Sets the palette generator algorithm.
                                Choices: OTFCD, QPUNM (default)
                                These only apply if there are more colors in the image
                                than there are color registers.
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

## API (not stable)
> [!WARNING]
> Do not rely on any method/variable that starts with an underscore as they are supposed to be private and can break at any moment without a warning. (Treat them like how you treat undefined behaviour in C++. 😉)

> [!TIP]
> Never set the `palette_generation_algorithm` unless you really have to, or just messing around. The default values should be better. This is likely to be more important in future versions.

```Python
print_image_from_path(path: str, *, register_count: int = 256, palette_generation_algorithm: PaletteGenerationAlgorithm = DEFAULT_PALETTE_GENERATION_ALGORITHM)

from_file_to_file(input_path: str, output_path: str, *, register_count: int = 256, palette_generation_algorithm: PaletteGenerationAlgorithm = DEFAULT_PALETTE_GENERATION_ALGORITHM)

img2sixels(image: List[List[Tuple[int, int, int, int]]], *, register_count: int = 256, palette_generation_algorithm: PaletteGenerationAlgorithm = DEFAULT_PALETTE_GENERATION_ALGORITHM) -> str
```

## TO-DO
- [x] ~~Multiprocessing~~
- [ ] Global interpreter lock detection to switch between multi-threading and multi-processing
- [ ] Dithering option
- [ ] Real-time SIXEL conversion for large images
- [ ] Ability to load and play videos (no, it won't make your "cat" play videos... probably)
- [ ] Auto select the optimal parameters based on the image or video, if not explicitly specified
- [ ] Play Bad Apple in real-time at full quality (must be done, one way or another)
- [ ] Lossless color output on all supported terminals (partially implemented and currently works on some terminals that allow register reuse)
- [ ] Get rid of as many hard dependencies as possibe

## Known issues
* Doesn't/shouldn't work on WebAssembly. (`concurrent.futures` library is not available, according to the Python documentation)
* No multiprocessing on mobile platforms. (`concurrent` library is not available)
* QPUNM is usually good enough but better options are definitely needed.
* Quality of OTFCD seems to be entirely dependent on a number that is too sensitive and the current way we calculate that number is just a very rough approximation.
* Pure Python seems to be too slow for real-time conversion of large detailed images.

## Q&A

**Q: Will you add this to PyPI?**

A: No, I can't bother with that for now. But you're free to do so, if you can maintain it. :)
