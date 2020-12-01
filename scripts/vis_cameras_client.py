import logging, argparse
import numpy as np
import cv2
import zmq
from time import sleep
from pickle import loads

from simple_zmq import SimpleZMQSubscriber


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', '-ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', '-p', type=str, default='5555')
    parser.add_argument('--topic', '-t', type=str, default='gym_cameras')
    args = parser.parse_args()

    sub = SimpleZMQSubscriber(args.ip, args.port, args.topic)

    window_names = set()
    logging.info('Running loop for {}'.format(args.topic))

    color_window_name = '{}/color'.format(args.topic)
    depth_window_name = '{}/depth'.format(args.topic)
    seg_window_name = '{}/seg'.format(args.topic)
    cv2.namedWindow(color_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(depth_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(seg_window_name, cv2.WINDOW_NORMAL)
    while True:
        ims_dict = sub.get()
        for key, im in ims_dict.items():
            if key == 'color':
                cv2.imshow(color_window_name, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            elif key == 'depth':
                cv2.imshow(depth_window_name, im / im.max())
            elif key == 'seg':
                cv2.imshow(seg_window_name, cv2.cvtColor((im / im.max() * 255).astype('uint8'), cv2.COLOR_GRAY2RGB))

        k = cv2.waitKey(1)
        if k == 27 or k == ord('q'):
            break

    cv2.destroyAllWindows()
