import argparse
import glob
import itertools
import os

from style_transfer import main as run

def main(args):
    content_images = glob.glob(os.path.join(args.content_dir, '*'))
    style_images = glob.glob(os.path.join(args.style_dir, '*'))

    pairs = itertools.product(content_images, style_images)

    for content, style in pairs:
        args.content_loc = content
        args.style_loc = style
        name1 = os.path.basename(content).split('.')[0]
        name2 = os.path.basename(style).split('.')[0]
        args.name = f'{name1}_{name2}.png'
        
        run(args)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--content-dir', required=True, help='path to content images')
    parser.add_argument('-s', '--style-dir', required=True, help='path to style images')
    parser.add_argument('-d', '--dimensions', nargs='+', type=int,
                        help='''height and width of output image in pixels. 
                                defaults to dimensions of content image if none is given.''')
    parser.add_argument('-o', '--output-dir', default='.', help='location to save output to')
    parser.add_argument('-i', '--interval', type=int,
                        help='''saves output to output directory every N epochs
                                where N is the value given''')
    parser.add_argument('-a', '--alpha', type=float, default=1, help='content weighting factor')
    parser.add_argument('-b', '--beta', type=float, default=1000000, help='style weighting factor')
    parser.add_argument('-t', '--tile', action='store_true', help='tile style image instead of resizing it')
    parser.add_argument('-e', '--epochs', default=1000, type=int, help='number of optimization steps')
    parser.add_argument('-l', '--learning-rate', type=float, help='learning rate of model')
    
    args = parser.parse_args()
    main(args)
