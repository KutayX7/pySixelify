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

def print_image_from_path(path: str, COLOR_REGISTER_COUNT: int = 256):
    try:
        import PIL.Image
    except:
        raise Exception("This function requires the `pillow` module. Install it with `pip install pillow`.")
    with PIL.Image.open(path) as image:
        image = image.convert("RGBA", colors=256)
        width, height = image.width, image.height
        flat_pixels = list(image.getdata())
        pixels_2d = [flat_pixels[i * width:(i + 1) * width] for i in range(height)]
        print(img2sixels(pixels_2d, COLOR_REGISTER_COUNT))


# Converts an image (2D lists of tuples, RGBA int[0, 255]) into an sixel image that can be printed.
# for black and white images, this should be near instant
# for grayscale images, this should take less than a second
# for color images, this can take up to a minute
def img2sixels(image: list[list[tuple[int, int, int, int]]], COLOR_REGISTER_COUNT: int = 256):
    width, height = len(image[0]), len(image)
    output = [f"\033P0;0;0q\"1;1;{width};{height}"]
    colors2str: dict[int, str] = {}
    colors: list[int] = []
    colorCounts: dict[int, int] = {}
    colorMap: dict[int, int] = {}
    color2RGB: dict[int, tuple[int, int, int]] = {}
    RGB2color: dict[tuple[int, int, int], int] = {}
    runlength2str = [str(i) for i in range(width+1)]
    mask2str = [chr(i + 63) for i in range(128)]
    
    sixel_cr = '$'
    sixel_crlf = '-'
    sixel_repeat = '!'
    
    def packRGB(r: int, g: int, b: int) -> int:
        if (r, g, b) in RGB2color:
            return RGB2color[(r, g, b)]
        result = int(r * 65536 + g * 256 + b)
        color2RGB[result] = (r, g, b)
        RGB2color[(r, g, b)] = result
        return result
    
    # pack image
    packed_image = []
    for y in range(height):
        packed_image.append([])
        for x in range(width):
            (r, g, b, a) = image[y][x]
            if a < 1:
                r, g, b = 0, 0, 0
            packed_image[y].append(packRGB(r, g, b))
    while (height % 6) > 0:
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
    
    # gnerate color palette
    # this is the most optimal way I found so far
    colors.sort(reverse=True, key=lambda e: colorCounts[e])
    RGBs = [color2RGB[c] for c in colors]
    for i in range(1, int(len(colors) ** 0.5)):
        j = i*i
        r, g, b = RGBs[i-1]
        rx, gx, bx = RGBs[i]
        ry, gy, by = RGBs[j]
        dx = abs(rx-r) + abs(gx-g) + abs(bx-x)
        dy = abs(ry-r) + abs(gy-g) + abs(by-x)
        if dy > dx:
            RGBs[i] = (ry, gy, by)
            RGBs[j] = (rx, gx, bx)
    colors = [RGB2color[t] for t in RGBs]
    alt_colors = colors[:COLOR_REGISTER_COUNT]
    alt_RGB = RGBs[:COLOR_REGISTER_COUNT]
    for i in range(min(len(colors), COLOR_REGISTER_COUNT)):
        color = colors[i]
        r, g, b = RGBs[i]
        colorMap[color] = color
        output.append(f'#{i};2;{int(r*100/255)};{int(g*100/255)};{int(b*100/255)}')
        colors2str[color] = f'#{i}'
    if len(colors) > COLOR_REGISTER_COUNT:
        for i in range(COLOR_REGISTER_COUNT, len(colors)):
            color = colors[i]
            r, g, b = RGBs[i]
            closest = (0,0,0)
            min_d = 4000000
            for r2, g2, b2 in alt_RGB:
                d = (r2-r)**2+(g2-g)**2+(b2-b)**2
                if d < min_d:
                    closest = (r2, g2, b2)
                    min_d = d
            colorMap[color] = RGB2color[closest]
    
    # convert colors according to the color palette
    for y in range(height):
        for x in range(width):
            packed_image[y][x] = colorMap[packed_image[y][x]]
    
    # flatten the packed image
    flattened_image = []
    for y in range(0, height, 6):
        for x in range(width):
            for i in range(y + 5, y - 1, -1):
                flattened_image.append(packed_image[i][x])
    
    # render
    for y in range(height//6):
        yw6 = y * width * 6
        colors_to_fill = alt_colors[:1]
        colors_to_fill_hashmap = {colors_to_fill[0]}
        max_y = min(y + 6, height) - 1
        range_height = range(max_y, y - 1, -1)
        start_indicies = {}
        end_indicies = {}
        
        # detect colors on the row
        for x in range(width * 6):
            c = flattened_image[yw6 + x]
            if c in colors_to_fill_hashmap:
                end_indicies[c] = x
            else:
                end_indicies[c] = x
                start_indicies[c] = x
                colors_to_fill_hashmap.add(c)
        start_indicies[colors_to_fill[0]] = start_indicies.get(colors_to_fill[0], width * 6)
        end_indicies[colors_to_fill[0]] = end_indicies.get(colors_to_fill[0], start_indicies[colors_to_fill[0]] - 1)
        colors_to_fill.pop()
        for c in colors_to_fill_hashmap:
            colors_to_fill.append(c)
        
        # draw row
        for c in colors_to_fill:
            start_index = int(start_indicies[c] / 6)
            end_index = int(end_indicies[c] / 6) + 1
            index = yw6 + start_index * 6
            output.append(colors2str[c])
            last_mask = 0
            run_length = start_index * (end_index > start_index)
            for x in range(start_index, end_index):
                mask  = (flattened_image[index    ] == c) * 32
                mask += (flattened_image[index + 1] == c) * 16
                mask += (flattened_image[index + 2] == c) * 8
                mask += (flattened_image[index + 3] == c) * 4
                mask += (flattened_image[index + 4] == c) * 2
                mask += (flattened_image[index + 5] == c)
                
                index += 6
                if last_mask == mask:
                    run_length += 1
                else:
                    maskStr = mask2str[last_mask]
                    if run_length > 9:
                        while run_length > 255: # for compatibility (max allowed repetitions is unknown)
                            output.append(f'!255{maskStr}')
                            run_length -= 255
                        output.append(f'!{runlength2str[run_length]}{maskStr}')
                    else:
                        output.append(maskStr * run_length)
                    last_mask = mask
                    run_length = 1
            maskStr = mask2str[last_mask]
            if run_length > 9:
                while run_length > 255: # compatibility
                    output.append(f'!255{maskStr}')
                    run_length -= 255
                output.append(f'!{runlength2str[run_length]}{maskStr}')
            else:
                output.append(maskStr * run_length)
            if end_index < width:
                output.append(chr(63) * (width - end_index))
            if start_index < width:
                output.append(sixel_cr)
        output.append(sixel_crlf)
    output.append("\033\\")
    return "".join(output)
