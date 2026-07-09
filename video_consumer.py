import numpy as np
from PIL import Image, ImageDraw
import datetime

_UTC = datetime.UTC


class VideoConsumer:
    def __init__(self, writer, font, frame_to_rgb, height, width, img_dtype, time_dtype, *, bin_size):
        self.writer = writer
        self.font = font
        self.frame_to_rgb = frame_to_rgb
        self.bin_size = bin_size
        self.frame_buffer = np.empty((bin_size, height, width), dtype=img_dtype)
        self.time_buffer = np.empty(bin_size, dtype=time_dtype)
        self.idx = 0

    def update(self, i, frame, time):

        self.frame_buffer[self.idx] = frame
        self.time_buffer[self.idx] = time
        self.idx += 1

        if self.idx == self.bin_size:
            write_video_frame(self.writer, self.frame_buffer, self.time_buffer, self.font, self.frame_to_rgb)
            self.idx = 0

    def finalize(self):
        print("finalzing video")
        if self.idx > 0:
            write_video_frame(
                self.writer, self.frame_buffer[: self.idx], self.time_buffer[: self.idx], self.font, self.frame_to_rgb
            )
            self.idx = 0


def write_video_frame(writer, frame_bin, time_bin, font, frame_to_rgb):
    if frame_bin.shape[0] == 1:  # avoid mean function if frames aren't binned
        frame = frame_bin[0]
        time_stamp = time_bin[0]
    else:
        frame = frame_bin.mean(axis=0)
        time_stamp = time_bin.mean()

    frame = frame_to_rgb(frame)
    frame = add_time_stamp(frame, time_stamp, font)
    writer.append_data(frame)


# def add_time_stamp(frame, time_stamp, font):
#     img = Image.fromarray(frame)
#     draw = ImageDraw.Draw(img)
#     text = unix_to_str(time_stamp)

#     x, y = 10, 10
#     draw.text((x - 1, y - 1), text, fill=(0, 0, 0), font=font)  # shadow
#     draw.text((x, y), text, fill=(255, 255, 255), font=font)  # text

#     return np.array(img)


def add_time_stamp(frame, time_stamp, font):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    text = unix_to_str(time_stamp)

    margin = 10
    # get text bounding box to know its height
    bbox = draw.textbbox((0, 0), text, font=font)
    text_height = bbox[3] - bbox[1]

    x = margin
    y = img.height - text_height - margin

    draw.text((x - 1, y - 1), text, fill=(0, 0, 0), font=font)  # shadow
    draw.text((x, y), text, fill=(255, 255, 255), font=font)  # text

    return np.array(img)


def unix_to_str(time_stamp):
    return datetime.datetime.fromtimestamp(float(time_stamp), _UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
