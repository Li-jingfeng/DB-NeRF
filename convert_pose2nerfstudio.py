import os

folder_path = "waymo_dynamic/waymo/processed/training/016/ego_pose"
filesnames = os.listdir(folder_path)




import json
with open('016_gtposes.txt', 'r') as f:
    lines = f.readlines()

json_file = {
    "camera_model": "OPENCV", 
    "fl_x": 2.056282369209058288e+03,  # focal length x
    "fl_y": 2.056282369209058288e+03,  # focal length y
    "cx": 9.395779800048162542e+02,    # principal point x
    "cy": 6.411030804525234998e+02,    # principal point y
    "w": 1920,       # image width
    "h": 1280,       # image height
}

# 解析每一行并转换为JSON格式
frames = []
for i, line in enumerate(lines):
    data = list(map(float, line.split()))
    transform_matrix = [
        data[0:4],
        data[4:8],
        data[8:12],
        [0.0, 0.0, 0.0, 1.0]  # 添加单位矩阵最后一行
    ]
    frame = {
        "file_path": "images/{:03d}_0.jpg".format(i),  # 假设文件路径是images/0001.jpg、images/0002.jpg等
        "transform_matrix": transform_matrix
    }
    frames.append(frame)

# 创建JSON对象
json_file['frames'] = frames

# 将JSON对象保存为文件
with open('waymo_dynamic/waymo/processed/training/016/transforms.json', 'w') as f:
    json.dump(json_file, f, indent=2)


# import json
# from PIL import Image
# with open('016_gtposes.txt', 'r') as f:
#     lines = f.readlines()

# json_file = {
#     "camera_model": "OPENCV_FISHEYE", 
#     "fl_x": 2.056282369209058288e+03,  # focal length x
#     "fl_y": 2.056282369209058288e+03,  # focal length y
#     "cx": 9.395779800048162542e+02,    # principal point x
#     "cy": 6.411030804525234998e+02,    # principal point y
#     "w": 1920,       # image width
#     "h": 1280,       # image height
# }

# # 解析每一行并转换为JSON格式
# frames = []
# for i, line in enumerate(lines):
#     image = Image.open('waymo_dynamic/waymo/processed/training/016/images/{:03d}_0.jpg'.format(i))
#     image.save('waymo_dynamic/waymo/processed/training/016/images_tensorf/{:03d}_0.png'.format(i))
#     data = list(map(float, line.split()))
#     transform_matrix = [
#         data[0:4],
#         data[4:8],
#         data[8:12],
#         [0.0, 0.0, 0.0, 1.0]  # 添加单位矩阵最后一行
#     ]
#     frame = {
#         "file_path": "images_tensorf/{:03d}_0".format(i),  # 假设文件路径是images/0001.jpg、images/0002.jpg等
#         "transform_matrix": transform_matrix
#     }
#     frames.append(frame)

# # 创建JSON对象
# json_file['frames'] = frames

# # 将JSON对象保存为文件
# with open('waymo_dynamic/waymo/processed/training/016/transforms_train.json', 'w') as f:
#     json.dump(json_file, f, indent=2)