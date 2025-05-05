# MIT License
# 
# Copyright (c) 2025 KutayX7
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# https://github.com/KutayX7/pySixelify

import argparse
from queue import Queue
from typing import List, Tuple, Dict, Literal, Any
from functools import cache

type PaletteGenerationAlgorithm = Literal['QPUNM', 'OTFCD']
type _RGBAImage = List[List[Tuple[int, int, int, int]]]
type _Color = int
type _ColorMap = Dict[_Color, _Color]
type _ColorCounter = Dict[_Color, int]
type _Color2strMap = Dict[_Color, str]
type _OutputStream = List[str]
type _1DImage = List[_Color]
type _2DImage = List[List[_Color]]
type _Mask = int
type _kwargs = Dict[str, Any]

DEFAULT_PALETTE_GENERATION_ALGORITHM: PaletteGenerationAlgorithm = 'QPUNM'

_mask2str = [chr(i + 63) for i in range(64)]
_runlength2str = ['!' + str(i) for i in range(256)]

def _from_file_to_RGBImage(file_path: str) -> _RGBAImage:
    try:
        import PIL.Image
    except:
        raise Exception("This function requires the `pillow` module. Install it with `pip install pillow`.")
    with PIL.Image.open(file_path) as image:
        image = image.convert("RGBA", colors=256)
        width, height = image.width, image.height
        flat_pixels = list(image.getdata()) # type: ignore
        return [flat_pixels[i * width:(i + 1) * width] for i in range(height)]

def print_image_from_path(path: str, *, register_count: int = 256, palette_generation_algorithm: PaletteGenerationAlgorithm = DEFAULT_PALETTE_GENERATION_ALGORITHM):
    image = _from_file_to_RGBImage(path)
    sixels = img2sixels(image, register_count=register_count, palette_generation_algorithm=palette_generation_algorithm)
    print(sixels)

def from_file_to_file(input_path: str, output_path: str, *, register_count: int = 256, palette_generation_algorithm: PaletteGenerationAlgorithm = DEFAULT_PALETTE_GENERATION_ALGORITHM):
    with open(output_path, 'wb') as file:
        image = _from_file_to_RGBImage(input_path)
        sixels = img2sixels(image, register_count=register_count, palette_generation_algorithm=palette_generation_algorithm)
        file.write(sixels.encode('utf-8'))

@cache
def _to_color(r: int, g: int, b: int) -> _Color:
    return (r << 16) + (g << 8) + b

@cache
def _to_RGB(color: _Color) -> tuple[int, int, int]:
    B = color % 256
    R = color >> 16
    G = (color >> 8) % 256
    return (R, G, B)

def _RGB_distance(r1: int, g1: int, b1: int, r2: int, g2: int, b2: int) -> int:
    return (r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2

def _color_distance(c1: _Color, c2: _Color) -> int:
    return _RGB_distance(*_to_RGB(c1), *_to_RGB(c2))

def _avg_colors(color_list: list[_Color]) -> _Color:
    tr: int = 0
    tg: int = 0
    tb: int = 0
    n = len(color_list)
    for color in color_list:
        r, g, b = _to_RGB(color)
        tr += r
        tg += g
        tb += b
    ar = int(tr / n)
    ag = int(tg / n)
    ab = int(tb / n)
    return _to_color(ar, ag, ab)

def _avg_RGBs(rgb_list: list[tuple[int, int, int]]) -> tuple[float, float, float]:
    tr: int = 0
    tg: int = 0
    tb: int = 0
    n = len(rgb_list)
    for r, g, b in rgb_list:
        tr += r
        tg += g
        tb += b
    return (tr/n, tg/n, tb/n)

def _round_RGB(r: float, g: float, b: float) -> tuple[int, int, int]:
    return (int(r + 0.5), int(g + 0.5), int(b + 0.5))

def _repeatMask(mask: _Mask, run_length: int, output: _OutputStream):
    s = _mask2str[mask]
    if run_length < 4:
        output.append(s * run_length)
        return
    while run_length > 255: # for compatibility (max allowed repetitions is unknown)
        output.append(f'!255{s}')
        run_length -= 255
    if run_length < 4:
        output.append(s * run_length)
        return
    output.append(f'!{run_length}{s}')

def _generate_color_map(colorCounts: _ColorCounter, register_count: int, algorithm: PaletteGenerationAlgorithm = DEFAULT_PALETTE_GENERATION_ALGORITHM) -> _ColorMap:
    if len(colorCounts) <= register_count:
        return {c: c for c in colorCounts}
    if algorithm == 'QPUNM':
        return _QPUNM(colorCounts, register_count)
    if algorithm == 'OTFCD':
        return _OTFCD(colorCounts, register_count)
    if algorithm == 'FPIR':
        return _FPIR(colorCounts, register_count)
    raise Exception(f'Unknown algorithm "{algorithm}"')

def _stringify_color_map(colorMap: _ColorMap, output: _OutputStream) -> _Color2strMap:
    result: _Color2strMap = dict()
    register_index = 0
    for color in list(colorMap.values()):
        if color not in result:
            r, g, b = _to_RGB(color)
            output.append(f'#{register_index};2;{int(r*100/255)};{int(g*100/255)};{int(b*100/255)}')
            result[color] = f'#{register_index}'
            register_index += 1
    return result

def _stringify_0_color_map(colorMap: _ColorMap) -> _Color2strMap:
    result: _Color2strMap = dict()
    for color in list(colorMap.values()):
        if color not in result:
            r, g, b = _to_RGB(color)
            result[color] = f'#0;2;{int(r*100/255)};{int(g*100/255)};{int(b*100/255)}'
    return result

def _remap_2d_image(image: _2DImage, colorMap: _ColorMap):
    height = len(image)
    width = len(image[0])
    for y in range(height):
        for x in range(width):
            image[y][x] = colorMap[image[y][x]]

# full palette, infinite registers
def _FPIR(colorCounts: _ColorCounter, register_count: int) -> _ColorMap:
    return {c: c for c in colorCounts}

# Quadratic Push Up, Nearest Match
def _QPUNM(colorCounts: _ColorCounter, register_count: int) -> _ColorMap:
    colorMap: _ColorMap = {}
    colors = [c for c in colorCounts]
    colors.sort(reverse=True, key=lambda e: colorCounts[e])
    RGBs = [_to_RGB(c) for c in colors]
    for i in range(1, int(len(colors) ** 0.5)):
        j = i*i
        r, g, b = RGBs[i-1]
        rx, gx, bx = RGBs[i]
        ry, gy, by = RGBs[j]
        dx = abs(rx-r) + abs(gx-g) + abs(bx-b)
        dy = abs(ry-r) + abs(gy-g) + abs(by-b)
        if dy > dx:
            RGBs[i] = (ry, gy, by)
            RGBs[j] = (rx, gx, bx)
    colors = [_to_color(r, g, b) for r, g, b in RGBs]
    Rs = [c[0] for c in RGBs]
    Gs = [c[1] for c in RGBs]
    Bs = [c[2] for c in RGBs]
    for i in range(min(len(colors), register_count)):
        color = colors[i]
        colorMap[color] = color
    if len(colors) > register_count:
        for i in range(register_count, len(colors)):
            color = colors[i]
            r = Rs[i]
            g = Gs[i]
            b = Bs[i]
            closest = 0
            min_diff = 200000
            for j in range(register_count):
                r2 = Rs[j]
                g2 = Gs[j]
                b2 = Bs[j]
                diff = (r2-r)**2+(g2-g)**2+(b2-b)**2
                if diff <= min_diff:
                    closest = j
                    min_diff = diff
            colorMap[color] = colors[closest]
    return colorMap

# type hell xD
# TODO: Fix type annotations
# TODO: Make this actually give decent results
# OctTree Fair Color Division
def _OTFCD(colorCounts: _ColorCounter, register_count: int) -> _ColorMap:
    colorMap: _ColorMap = {}
    RGBs = [_to_RGB(color) for color in colorCounts]
    root = []

    def is_leaf_node(node: list[object]):
        if len(node):
            return isinstance(node[0], tuple)
        return True

    def is_divisible(node: list[object]):
        return is_leaf_node(node) and len(node) > register_count * 8
    
    def divide(node, depth=1): # type: ignore
        if depth > 3:
            return depth
        ar, ag, ab = _avg_RGBs(node) # type: ignore
        buckets = [[] for _ in range(8)] # type: ignore
        for r, g, b in node: # type: ignore
            index = 4 if r >= ar else 0
            if g >= ag:
                index += 2
            if b >= ab:
                index += 1
            buckets[index].append((r, g, b)) # type: ignore
        node.clear() # type: ignore
        node.extend(buckets) # type: ignore
        node.append(_round_RGB(ar, ag, ab)) # type: ignore
        max_depth = depth
        for child in node[:8]: # type: ignore
            if is_divisible(child): # type: ignore
                max_depth = max(max_depth, divide(child, depth+1)) # type: ignore
            if is_leaf_node(child): # type: ignore
                if len(child): # type: ignore
                    child.append(_round_RGB(*_avg_RGBs(child))) # type: ignore
                else:
                    child.append(node[8]) # type: ignore
        return max_depth
    
    node_queue = Queue() # type: ignore
    root.extend(RGBs) # type: ignore
    divide(root) # type: ignore
    node_queue.put_nowait(root) # type: ignore
    remaining_register_count = register_count
    backup_color = 0 # type: ignore

    while node_queue.qsize():
        node = node_queue.get_nowait() # type: ignore
        ar, ag, ab = node[len(node)-1] # type: ignore
        avgc = _to_color(ar, ag, ab) # type: ignore
        if is_leaf_node(node): # type: ignore
            if avgc not in colorMap:
                if remaining_register_count and (len(node) > 1): # type: ignore
                    colorMap[avgc] = avgc
                    remaining_register_count -= 1
                    backup_color = avgc # type: ignore
                else:
                    avgc = backup_color
            for i in range(len(node)-1): # type: ignore
                r, g, b = node[i] # type: ignore
                color = _to_color(r, g, b) # type: ignore
                if color not in colorMap:
                    colorMap[color] = colorMap[avgc]
        else:
            for child in node[:8]: # type: ignore
                node_queue.put_nowait(child) # type: ignore
            if avgc in colorMap:
                backup_color = colorMap[avgc]
    return colorMap

def _render_sixels(image: _1DImage, width: int, color2str: _Color2strMap, output: _OutputStream):
    height = len(image) // width
    for y in range(height//6):
        yw6 = y * width * 6
        colors_to_fill: list[int] = list()
        colors_to_fill_set: set[int] = set()
        start_indicies: dict[int, int] = dict()
        end_indicies: dict[int, int] = dict()
        
        # detect colors on the row
        for x in range(width * 6):
            c = image[yw6 + x]
            if c in colors_to_fill_set:
                end_indicies[c] = x
            else:
                end_indicies[c] = x
                start_indicies[c] = x
                colors_to_fill_set.add(c)
        
        for c in colors_to_fill_set:
            colors_to_fill.append(c)
        
        worst_rl = 0
        worst_color = colors_to_fill[0]
        for c in start_indicies:
            rl = 1 + end_indicies[c] - start_indicies[c]
            if rl > worst_rl:
                worst_rl = rl
                worst_color = c
        
        # early row fill
        c = worst_color
        output.append(color2str[c])
        _repeatMask(63, width, output)
        output.append('$')

        # draw row
        for c in colors_to_fill:
            if c == worst_color:
                continue
            start_index = int(start_indicies[c] / 6)
            end_index = int(end_indicies[c] / 6) + 1
            index = yw6 + start_index * 6
            output.append(color2str[c])
            last_mask = 0
            run_length = start_index * (end_index > start_index)
            for x in range(start_index, end_index):
                mask  = (image[index    ] == c) * 32
                mask += (image[index + 1] == c) * 16
                mask += (image[index + 2] == c) * 8
                mask += (image[index + 3] == c) * 4
                mask += (image[index + 4] == c) * 2
                mask += (image[index + 5] == c)
                
                index += 6
                if last_mask == mask:
                    run_length += 1
                else:
                    _repeatMask(last_mask, run_length, output)
                    last_mask = mask
                    run_length = 1
            _repeatMask(last_mask, run_length, output)
            if end_index < width:
                _repeatMask(0, width - end_index, output)
            if start_index < width:
                output.append('$')
        output.append('-')

# Converts an image (2D lists of tuples, RGBA int[0, 255]) into an sixel image that can be printed.
# for black and white images, this should be near instant
# for grayscale images, this should take less than a few seconds
# for colored images, this can take up to a minute or two (depending on the image size, the amount of different colors in the source image and the amount of required color registers)
# EXPERIMENTAL: setting `register_count` argument to anything less than 2, outputs a full color image
def img2sixels(image: _RGBAImage, *, register_count: int = 256, palette_generation_algorithm: PaletteGenerationAlgorithm = DEFAULT_PALETTE_GENERATION_ALGORITHM) -> str:
    if register_count < 2:
        return _img2sixels_full_color(image)
    width, height = len(image[0]), len(image)
    output = [f"\033P0;0;0q\"1;1;{width};{height}"]
    colors2str: dict[int, str] = {}
    colors: list[int] = []
    colorCounts: dict[int, int] = {}
    colorMap: dict[int, int] = {}
    
    # pack image
    packed_image: list[list[int]] = []
    for y in range(height):
        packed_image.append([])
        for x in range(width):
            (r, g, b, a) = image[y][x]
            if a < 1:
                r, g, b = int(r * a), int(g * a), int(b * a)
            packed_image[y].append(_to_color(r, g, b))
    while (height % 6):
        height = height + 1
        packed_image.append([0] * width)
    
    #count colors
    for y in range(height):
        for x in range(width):
            color = packed_image[y][x]
            if color in colorCounts:
                colorCounts[color] += 1
            else:
                colors.append(color)
                colorCounts[color] = 1
    colorCounts[0] = 2**31
    
    # color palette
    colorMap = _generate_color_map(colorCounts, register_count, palette_generation_algorithm)
    colors2str = _stringify_color_map(colorMap, output)
    
    # convert colors according to the color palette
    _remap_2d_image(packed_image, colorMap)
    
    # flatten the packed image
    flattened_image: list[int] = []
    for y in range(0, height, 6):
        for x in range(width):
            for i in range(y + 5, y - 1, -1):
                flattened_image.append(packed_image[i][x])
    
    # render
    _render_sixels(flattened_image, width, colors2str, output)
    output.append("\033\\")
    return "".join(output)

# !EXPERIMENTAL!
# this is supposed to render a full color sixel image without any register limits
# a single register is reused for every color
# may not work with every sixel terminal
# may not work if the image is too big or has too many details
# this is likely to work slower than the normal method (but exceptions can happen)
# TODO: find a way to compress the output even further (which is necessary for this to work better with complex images)
# TODO: find edge cases (if any) and make sure they don't cause issues
def _img2sixels_full_color(image: _RGBAImage) -> str:
    width, height = len(image[0]), len(image)
    output = [f"\033P0;0;0q\"1;1;{width};{height}#0"]
    color2RGB: dict[int, tuple[int, int, int]] = {0: (0, 0, 0)}
    RGB2color: dict[tuple[int, int, int], int] = {(0, 0, 0): 0}
    
    def packRGB(r: int, g: int, b: int) -> int:
        if (r, g, b) in RGB2color:
            return RGB2color[(r, g, b)]
        result = int(r * 65536 + g * 256 + b)
        color2RGB[result] = (r, g, b)
        RGB2color[(r, g, b)] = result
        return result

    # pack image
    packed_image: list[list[int]] = []
    for y in range(height):
        packed_image.append([])
        for x in range(width):
            (r, g, b, a) = image[y][x]
            if a < 1:
                r, g, b = int(r * a), int(g * a), int(b * a)
            packed_image[y].append(packRGB(r, g, b))
    while (height % 6):
        height = height + 1
        packed_image.append([0] * width)
    
    # flatten the packed image
    flattened_image: list[int] = []
    for y in range(0, height, 6):
        for x in range(width):
            for i in range(y + 5, y - 1, -1):
                flattened_image.append(packed_image[i][x])
    
    # render
    _render_sixels(flattened_image, width, _stringify_0_color_map(_FPIR({c: 1 for c in flattened_image}, 1)), output)
    output.append("\033\\")
    return "".join(output)


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('filename', default='', nargs='?')
    _parser.add_argument('-r', '-cr', '--register-count', '--color-register-count', type=int, default=256, required=False, choices=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    _parser.add_argument('-o', '--output-file', default='', required=False)
    _parser.add_argument('-s', '--silent', action='store_true', required=False, default=False)
    _parser.add_argument('-p', '-pg', '--palette', '--palette-generator', choices=['QPUNM', 'OTFCD'], required=False, default=DEFAULT_PALETTE_GENERATION_ALGORITHM)
    _namespace = _parser.parse_args()
    if _namespace.filename:
        import os
        if not _namespace.silent:
            if _namespace.register_count == 1:
                print("WARNING: You're using an experimental feature `register_count==1`. If things go wrong, please report them at \"https://github.com/KutayX7/pySixelify/issues\"")
        if _namespace.output_file:
            from_file_to_file(os.path.abspath(_namespace.filename), os.path.abspath(_namespace.output_file), register_count=_namespace.register_count, palette_generation_algorithm=_namespace.palette)
        else:
            print_image_from_path(os.path.abspath(_namespace.filename), register_count=_namespace.register_count, palette_generation_algorithm=_namespace.palette)
