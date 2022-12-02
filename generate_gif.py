import glob
from os.path import join as joinp
from contextlib import ExitStack
from PIL import Image
from config import TRAIN_PLOTS_PATH


# filepaths
fp_in = joinp(TRAIN_PLOTS_PATH, f"gm_plot_*.png")
fp_out = "cover.gif"


# use exit stack to automatically close opened images
with ExitStack() as stack:

    # lazily load images
    imgs = (stack.enter_context(Image.open(f))
            for f in sorted(
                glob.glob(fp_in),
                key=lambda s: int(s.split('_')[-1][:-4]))
            if int(f.split('_')[-1][:-4]) % 50 == 0 or int(f.split('_')[-1][:-4]) < 200)

    # extract  first image from iterator
    img = next(imgs)

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)