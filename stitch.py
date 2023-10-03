from PIL import Image
import os

def natural_sort(lst):
    """
    Sort a list in a human-friendly order.
    """
    import re

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    return sorted(lst, key=natural_keys)

def split_image_into_tiles(image_path, tile_dir, tile_size=128):
    image = Image.open(image_path)
    width, height = image.size
    tile_count = 0

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = image.crop((x, y, x + tile_size, y + tile_size))

            if tile.size == (tile_size, tile_size):
                tile_count += 1
                tile.save(tile_dir + f"{tile_count}.png")

    return tile_count

def stitch_tiles_together(tile_dir, output_path):
    tiles = natural_sort([f for f in os.listdir(tile_dir) if f.endswith('.png')])
    print(tiles)
    tile_images = [Image.open(os.path.join(tile_dir, tile)) for tile in tiles]

    image_width = 128 * 18
    image_height = 128 * 12
    result_image = Image.new("RGB", (int(image_width), int(image_height)))

    count = 0

    
    for j in range(12):
        for i in range(18):
            tile_image = tile_images[count]
            count += 1
            x = i*128
            y = j*128
            result_image.paste(tile_image, (x, y))
    result_image.save(output_path)

if __name__ == "__main__":
    image_path = "./facebook-twitter/twitter/1.jpg"  # Replace with your image file path
    tile_dir = "./extracted/"  # Directory to store the tiles
    # image_path = "./c2pa-images/c2pa-1.jpg"  # Replace with your image file path
    # tile_dir = "./embed/"
    os.makedirs(tile_dir, exist_ok=True)
    
    tile_count = split_image_into_tiles(image_path, tile_dir, tile_size=128)
    print(f"Split {tile_count} tiles.")



#stitch

    # embedded = "./embedded/"

    # stitch_tiles_together(embedded, "stitched_image.png")
    # print("Tiles stitched back together as stitched_image.png")