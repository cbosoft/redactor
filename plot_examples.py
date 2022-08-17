from glob import glob

from matplotlib import pyplot as plt
import cv2

from redactor.redact import Redactor


if __name__ == '__main__':
    redactor = Redactor()

    imgs = sorted(glob('example_data/input_*'))

    for fn in imgs:
        i = int(fn.split('_')[-1][:-4])
        redacted = redactor.redact_image(fn)

        fig, (oax, rax) = plt.subplots(ncols=2, figsize=(15, 5))

        arrow_ax_w, arrow_ax_h = 0.05, 0.5
        arrow_ax = plt.gcf().add_axes((0.5 - arrow_ax_w/2, 0.5 - arrow_ax_h/2, arrow_ax_w, arrow_ax_h))
        arrow_ax.annotate(
            "", xy=(1, 0.5), xytext=(0, 0.5),
            arrowprops=dict(width=30, headwidth=60, color='k'))

        list(map(lambda ax: ax.axis('off'), [oax, rax, arrow_ax]))

        orig_img = cv2.imread(fn, cv2.IMREAD_COLOR)
        oax.set_title('Input')
        oax.imshow(orig_img[..., ::-1])
        rax.set_title('Redacted')
        rax.imshow(redacted[..., ::-1])
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'example_data/redacted_{i}.png')
        plt.close()
