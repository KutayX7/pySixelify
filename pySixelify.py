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
import concurrent.futures
from queue import Queue
from typing import List, Set, Tuple, Dict, Literal
from itertools import repeat

type PaletteGenerationAlgorithm = Literal['QPUNM', 'OTFCD']
type _RGBTuple = Tuple[int, int, int]
type _RGBFloatTuple = Tuple[float, float, float]
type _RGBATuple = Tuple[int, int, int, int]
type _RGBImage = List[List[_RGBTuple]]
type _RGBAImage = List[List[_RGBATuple]]
type _Color = int
type _ColorList = List[_Color]
type _ColorSet = Set[_Color]
type _ColorMap = Dict[_Color, _Color]
type _ColorCounter = Dict[_Color, int]
type _Color2strMap = Dict[_Color, str]
type _Color2bytesMap = Dict[_Color, bytes]
type _OutputStream = List[str]
type _1DImage = List[_Color]
type _1DImageBytes = bytearray
type _2DImage = List[List[_Color]]
type _Mask = int

type _OctreeColorNode = list[_OctreeColorNode|_RGBTuple]

DEFAULT_PALETTE_GENERATION_ALGORITHM: PaletteGenerationAlgorithm = 'QPUNM'
_QPUNM_CHUNK_SIZE: int = 4096
_RENDERING_CHUNK_SIZE: int = 1

_mask2byte = [bytes([i + 63]) for i in range(64)]
_mask2str = [chr(i + 63) for i in range(64)]
_runlength2bytes = [b'!' + str(i).encode(encoding='ascii') for i in range(256)]

def _get_executor() -> concurrent.futures.Executor:
    try:
        return concurrent.futures.ProcessPoolExecutor()
    except:
        return concurrent.futures.ThreadPoolExecutor()
_executor = _get_executor()

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

def _to_color(r: int, g: int, b: int) -> _Color:
    return (r << 16) + (g << 8) + b

def _to_RGB(color: _Color) -> _RGBTuple:
    B = color % 256
    R = color >> 16
    G = (color >> 8) % 256
    return (R, G, B)

def _get_R(color: _Color) -> int:
    return color >> 16
def _get_G(color: _Color) -> int:
    return (color >> 8) % 256
def _get_B(color: _Color) -> int:
    return color % 256

def _round_RGB(r: float, g: float, b: float) -> _RGBTuple:
    return (int(r + 0.5), int(g + 0.5), int(b + 0.5))

def _repeat_mask(mask: _Mask, run_length: int, output: _OutputStream):
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

def _bytify_color_map(colorMap: _ColorMap, output: _OutputStream) -> _Color2bytesMap:
    result: _Color2bytesMap = dict()
    register_index = 0
    remapping: dict[_Color, int] = dict()
    for color in list(set(colorMap.values())):
        r, g, b = _to_RGB(color)
        output.append(f'#{register_index};2;{int(r*100/255)};{int(g*100/255)};{int(b*100/255)}')
        result[register_index] = f'#{register_index}'.encode(encoding='ascii')
        remapping[color] = register_index
        register_index += 1
    for color in list(colorMap.keys()):
        to_color = colorMap[color]
        if to_color in remapping:
            colorMap[color] = remapping[to_color]
    for color in list(remapping.keys()):
        colorMap[color] = remapping[color]
    for i in range(256):
        if i not in colorMap:
            colorMap[i] = 0
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
def _FPIR(colorCounts: _ColorCounter, register_count: int = 256) -> _ColorMap:
    return {c: c for c in colorCounts}

def _find_closest(color: int, Rs: list[int], Gs: list[int], Bs: list[int]) -> Tuple[_Color, _Color]:
    r, g, b = _to_RGB(color)
    closest = 0
    min_diff = 400000
    for j in range(len(Bs)):
        r2 = Rs[j]
        g2 = Gs[j]
        b2 = Bs[j]
        rd = r2-r
        gd = g2-g
        bd = b2-b
        diff = rd*rd + gd*gd + bd*bd
        if diff <= min_diff:
            closest = j
            min_diff = diff
    return color, closest

# Quadratic Push Up, Nearest Match
def _QPUNM(color_counts: _ColorCounter, register_count: int) -> _ColorMap:
    colorMap: _ColorMap = {}
    colors: _ColorList = [c for c in color_counts]
    colors.sort(reverse=True, key=lambda e: color_counts[e])
    Rs: list[int] = [_get_R(c) for c in colors]
    Gs: list[int] = [_get_G(c) for c in colors]
    Bs: list[int] = [_get_B(c) for c in colors]
    for i in range(1, int(len(colors) ** 0.5)):
        j = i*i
        r = Rs[i-1]
        g = Gs[i-1]
        b = Bs[i-1]
        rx = Rs[i]
        gx = Gs[i]
        bx = Bs[i]
        ry = Rs[j]
        gy = Gs[j]
        by = Bs[j]
        dx = abs(rx-r) + abs(gx-g) + abs(bx-b)
        dy = abs(ry-r) + abs(gy-g) + abs(by-b)
        if dy > dx:
            Rs[i] = ry
            Gs[i] = gy
            Bs[i] = by
            Rs[j] = rx
            Gs[j] = gx
            Bs[j] = bx
            c = colors[i]
            colors[i] = colors[j]
            colors[j] = c
    for i in range(min(len(colors), register_count)):
        color = colors[i]
        colorMap[color] = color
    if len(colors) > register_count:
        for color, closest in _executor.map(_find_closest, colors[register_count:], repeat(Rs[:register_count]), repeat(Gs[:register_count]), repeat(Bs[:register_count]), chunksize=_QPUNM_CHUNK_SIZE):
            colorMap[color] = colors[closest]
    return colorMap

def _avg_RGBs(rgb_list: _OctreeColorNode) -> _RGBFloatTuple:
    tr: int = 0
    tg: int = 0
    tb: int = 0
    n = len(rgb_list)
    for r, g, b in rgb_list:
        tr += r # type: ignore
        tg += g # type: ignore
        tb += b # type: ignore
    return (tr/n, tg/n, tb/n) # type: ignore

# OctTree Fair Color Division
def _OTFCD(colorCounts: _ColorCounter, register_count: int) -> _ColorMap:
    colorMap: _ColorMap = {}
    RGBs: List[_RGBTuple] = [_to_RGB(color) for color in colorCounts]
    root: _OctreeColorNode = []
    node_queue: Queue[_OctreeColorNode] = Queue()
    remaining_color_queue: Queue[_Color] = Queue()
    remaining_register_count: int = register_count
    backup_color: _Color = max(colorCounts, key=lambda color: colorCounts[color])
    color_count: int = len(colorCounts)
    division_threshold: int = int(color_count * 256 * 256 * 2 / register_count ** 3)

    def unpack_node(node: _OctreeColorNode|_RGBFloatTuple) -> _RGBTuple:
        r, g, b = node
        return r, g, b # type: ignore

    def is_leaf_node(node: _OctreeColorNode) -> bool:
        if len(node):
            return isinstance(node[0], tuple)
        return True

    def is_divisible(node: _OctreeColorNode) -> bool:
        return is_leaf_node(node) and len(node) > division_threshold
    
    def divide(node: _OctreeColorNode, depth: int = 1) -> int:
        if depth > 3:
            return depth
        ar, ag, ab = _avg_RGBs(node)
        buckets: List[_OctreeColorNode] = [[] for _ in range(8)]
        for item in node:
            r, g, b = unpack_node(item)
            index = 4 if r >= ar else 0
            if g >= ag:
                index += 2
            if b >= ab:
                index += 1
            buckets[index].append((r, g, b))
        node.clear()
        node.extend(buckets)
        node.append(_round_RGB(ar, ag, ab))
        max_depth = depth
        for child in buckets:
            if is_divisible(child):
                max_depth = max(max_depth, divide(child, depth+1))
            if is_leaf_node(child):
                if len(child):
                    child.append(_round_RGB(*_avg_RGBs(child)))
                else:
                    child.append(node[8])
        return max_depth
    
    root.extend(RGBs)
    divide(root)
    node_queue.put_nowait(root)

    while node_queue.qsize():
        node: _OctreeColorNode = node_queue.get_nowait()
        ar, ag, ab = unpack_node(node[len(node)-1])
        avgc = _to_color(ar, ag, ab)
        if is_leaf_node(node):
            if avgc not in colorMap:
                if remaining_register_count > 0 and (len(node) > 1):
                    colorMap[avgc] = avgc
                    remaining_register_count -= 1
                    backup_color = avgc
                else:
                    avgc = backup_color
            for child in node[:-1]:
                color = _to_color(*unpack_node(child))
                if color not in colorMap:
                    colorMap[color] = colorMap[avgc]
                    remaining_color_queue.put_nowait(color)
        else:
            if remaining_register_count == 0:
                for child in node[:8]:
                    if isinstance(child, list):
                        child[len(child)-1] = _to_RGB(avgc)
            for child in node[:8]:
                if isinstance(child, list):
                    node_queue.put_nowait(child)
            if avgc in colorMap:
                backup_color = colorMap[avgc]

    if remaining_register_count > register_count:
        while remaining_register_count > 0 and remaining_color_queue.qsize() > 0:
            color: _Color = remaining_color_queue.get_nowait()
            if colorMap[color] != color:
                colorMap[color] = color
                remaining_register_count -= 1
    return colorMap

def _repeat_mask_bytes(mask: int, run_length: int) -> bytearray:
    s: bytes = _mask2byte[mask]
    result: bytearray = bytearray()
    if run_length < 4:
        result += s * run_length
        return result
    while run_length > 255: # for compatibility (max allowed repetitions is unknown)
        result += b'!255' + s
        run_length -= 255
    if run_length < 4:
        result += s * run_length
        return result
    result += _runlength2bytes[run_length] + s
    return result

def _render_row(y: int, image: _1DImageBytes, width: int, height: int, color2bytes: _Color2bytesMap) -> bytearray:
    row_out: bytearray = bytearray()
    yw6: int = y * width * 6
    colors_to_fill: _ColorList = list()
    colors_to_fill_set: _ColorSet = set()
    start_indicies: dict[_Color, int] = dict()
    end_indicies: dict[_Color, int] = dict()
    
    # detect colors on the row
    for x in range(width * 6):
        c: _Color = image[yw6 + x]
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
        rl: int = 1 + end_indicies[c] - start_indicies[c]
        if rl > worst_rl:
            worst_rl = rl
            worst_color = c
    
    # early row fill
    c: _Color = worst_color
    assert(c in color2bytes)
    row_out += color2bytes[c]
    row_out += _repeat_mask_bytes(63, width)
    row_out += b'$'

    # draw row
    for c in colors_to_fill:
        if c == worst_color:
            continue
        start_index: int = int(start_indicies[c] / 6)
        end_index: int = int(end_indicies[c] / 6) + 1
        index: int = yw6 + start_index * 6
        row_out += color2bytes[c]
        last_mask: _Mask = 0
        run_length: int = start_index * (end_index > start_index)
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
                row_out += _repeat_mask_bytes(last_mask, run_length)
                last_mask = mask
                run_length = 1
        row_out += _repeat_mask_bytes(last_mask, run_length)
        if end_index < width:
            row_out += _repeat_mask_bytes(0, width - end_index)
        if start_index < width:
            row_out += b'$'
    row_out += b'-'
    return row_out

def _render_sixels(image: _1DImageBytes, width: int, color2bytes: _Color2bytesMap) -> bytearray:
    height: int = len(image) // width
    y_array: list[int] = list(range(height//6))
    sixel_out = bytearray()
    for row in _executor.map(_render_row, y_array, repeat(image), repeat(width), repeat(height), repeat(color2bytes), chunksize=_RENDERING_CHUNK_SIZE):
        sixel_out += row
    return sixel_out

def _render_full(image: _1DImage, width: int, color2str: _Color2strMap, output: _OutputStream):
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
        _repeat_mask(63, width, output)
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
                    _repeat_mask(last_mask, run_length, output)
                    last_mask = mask
                    run_length = 1
            _repeat_mask(last_mask, run_length, output)
            if end_index < width:
                _repeat_mask(0, width - end_index, output)
            if start_index < width:
                output.append('$')
        output.append('-')

# Converts an image (2D lists of tuples, RGBA int[0, 255]) into an sixel image that can be printed.
def img2sixels(image: _RGBAImage, *, register_count: int = 256, palette_generation_algorithm: PaletteGenerationAlgorithm = DEFAULT_PALETTE_GENERATION_ALGORITHM) -> str:
    assert(register_count == int(register_count))
    assert(register_count > 0)
    assert(register_count <= 256)
    if register_count == 1:
        return _img2sixels_full_color(image)
    width: int = len(image[0])
    height: int = len(image)
    output: _OutputStream = [f"\033P0;0;0q\"1;1;{width};{height}"]
    colors: _ColorList = []
    colorCounts: _ColorCounter = {}
    
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
    colorMap: _ColorMap = _generate_color_map(colorCounts, register_count, palette_generation_algorithm)
    colors2bytes: _Color2bytesMap = _bytify_color_map(colorMap, output)
    
    # convert colors according to the color palette
    _remap_2d_image(packed_image, colorMap)
    
    # flatten the packed image
    flattened_image: _1DImageBytes = bytearray()
    for y in range(0, height, 6):
        for x in range(width):
            for i in range(y + 5, y - 1, -1):
                flattened_image.append(packed_image[i][x])
    
    # render
    result = _render_sixels(flattened_image, width, colors2bytes)
    output.append(result.decode(encoding='ascii'))
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
    flattened_image: _1DImage = []
    for y in range(0, height, 6):
        for x in range(width):
            for i in range(y + 5, y - 1, -1):
                flattened_image.append(packed_image[i][x])
    
    # render
    _render_full(flattened_image, width, _stringify_0_color_map(_FPIR({c: 1 for c in flattened_image}, 1)), output)
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
