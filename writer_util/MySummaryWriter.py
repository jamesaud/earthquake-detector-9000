from PIL import Image
import os
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import io
from torchvision.transforms import ToTensor

class MySummaryWriter(SummaryWriter):

    def __init__(self, path, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self.path = path
        self.image_path = os.path.join(self.path, 'images/')
        self.figures = []


    def add_plt_figure(self, tag, figure, global_step=None):
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        img = ToTensor()(Image.open(buf))
        buf.close()
        self.add_image(tag=tag, img_tensor=img, global_step=global_step)

    def figure_to_image(self, figure):
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        return img

    def combine_images_vertical(self, images):
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        total_height = sum(heights)

        image = Image.new('RGB', (max_width, total_height))
        y_offset = 0

        for im in images:
            width, height = im.size
            image.paste(im, (0, y_offset))
            y_offset += height

        return image

    def combine_images_horizontal(self, images):
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)

        image = Image.new('RGB', (total_width, max_height))
        x_offset = 0

        for im in images:
            width, height = im.size
            image.paste(im, (x_offset, 0))
            x_offset += width

        return image

    def add_pil_image(self, tag, image, global_step=None):
        image = ToTensor()(image)
        return self.add_image(tag, image, global_step=global_step)

