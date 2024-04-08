import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time


data_path = r'/amax/DATA/dataset/FELT/test'
save_path = r'/amax/DATA/dataset/FELT/fusion_test'
if not os.path.exists(save_path):
    os.mkdir(save_path)

filenames = sorted([f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))])


def is_directory_empty(directory_path):
    return len(os.listdir(directory_path)) == 0


file_count = len(filenames)

for count, filename in enumerate(filenames, start=1):
    print(f'[{count} / {file_count}]开始处理：', filename)
    aps_sequence_path = os.path.join(data_path, filename, filename + "_aps")
    dvs_sequence_path = os.path.join(data_path, filename, filename + "_dvs")
    
    aps_count = len(os.listdir(aps_sequence_path))
    fusion_count = len(os.listdir(os.path.join(save_path, filename + '_fusion', filename + "_fusion_aps")))


    if not os.path.exists(os.path.join(save_path, filename + '_fusion', filename + "_fusion_aps" )):
        os.makedirs(os.path.join(save_path, filename + '_fusion', filename + "_fusion_aps"))
        
    if aps_count == fusion_count:
        print(f"{aps_count}={fusion_count} ==>> Skip this video : {filename}")
        continue
    
    
    # if not is_directory_empty(os.path.join(save_path, filename + '_fusion', filename + "_fusion_aps")):
    #     print(f"==>> Skip this video : {filename}")
    #     continue

    aps_count = len(os.listdir(aps_sequence_path))

    for frame_id in range(0, aps_count):
        if os.path.exists(os.path.join(aps_sequence_path, 'frame{:04}.png'.format(frame_id))):
            rgb_frame_path = os.path.join(aps_sequence_path, 'frame{:04}.png'.format(frame_id))
        else:
            rgb_frame_path = os.path.join(aps_sequence_path, 'frame{:04}.bmp'.format(frame_id))

        if os.path.exists(os.path.join(dvs_sequence_path, 'frame{:04}.png'.format(frame_id))):
            event_frame_path = os.path.join(dvs_sequence_path, 'frame{:04}.png'.format(frame_id))
        else:
            event_frame_path = os.path.join(dvs_sequence_path, 'frame{:04}.bmp'.format(frame_id))

        fusion_save_path = os.path.join(save_path, filename + '_fusion', filename + "_fusion_aps", 'frame{:04}.bmp'.format(frame_id))

        rgb_frame = Image.open(rgb_frame_path).convert("RGBA")
        event_frame = Image.open(event_frame_path).convert("RGBA")
        # rgb_frame = Image.open(rgb_frame_path)
        # event_frame = Image.open(event_frame_path)
        data = event_frame.getdata()

        new_data = []
        for item in data:
            # 如果像素点不是白色，将其保留
            if item[:3] != (255, 255, 255):
                new_data.append(item)
            else:
                # 如果是白色像素点，设置为透明
                new_data.append((255, 255, 255, 0))

        event_frame.putdata(new_data)

        fusion_frame = Image.new("RGBA", rgb_frame.size)

        for x in range(rgb_frame.width):
            for y in range(rgb_frame.height):
                bg_pixel = rgb_frame.getpixel((x, y))
                fg_pixel = event_frame.getpixel((x, y))

                # 如果前景图像对应像素点是透明的，则使用背景图像的颜色
                if fg_pixel[3] == 0:
                    fusion_frame.putpixel((x, y), bg_pixel)
                else:
                    # 混合前景和背景颜色
                    blended_pixel = (
                        int((fg_pixel[0] + bg_pixel[0]) / 2),
                        int((fg_pixel[1] + bg_pixel[1]) / 2),
                        int((fg_pixel[2] + bg_pixel[2]) / 2),
                        255
                    )
                    fusion_frame.putpixel((x, y), blended_pixel)

        # fusion_frame = Image.blend(rgb_frame, event_frame, alpha=0.2)
        # fusion_frame.show()

        fusion_frame.save(fusion_save_path)

        print(f'[{count} / {file_count}] :{filename}','save success', 'frame{:04}.bmp'.format(frame_id))


