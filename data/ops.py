from PIL import ImageOps
from torchvision.transforms import functional

def resize_and_padding(image, target_shape):
    pil_image = functional.to_pil_image(image)

    scale = min(target_shape[0] / pil_image.height, target_shape[1] / pil_image.width)

    resize_width = int(pil_image.width * scale)
    resize_height = int(pil_image.height * scale)

    pil_resize_image = pil_image.resize((resize_width, resize_height))
    padding_bottom, padding_right = target_shape[0] - resize_height, target_shape[1] - resize_width
    pil_image = ImageOps.expand(pil_resize_image, (0, 0, padding_right, padding_bottom), fill=0)

    return functional.to_tensor(pil_image), (padding_bottom, padding_right), (pil_image.height, pil_image.width)