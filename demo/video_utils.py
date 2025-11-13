import os
import cv2
import numpy as np


def read_video(video_path):
    capture = cv2.VideoCapture(video_path)
    frame_count = 0
    video_frames = []
    gsp_video_frames = []
    if capture.isOpened():
        while True:
            ret, img = capture.read()  # img 就是一帧图片
            if not ret:
                print('Video reading complete.')
                fps = capture.get(cv2.CAP_PROP_FPS)  # 获取当前版本opencv的FPS
                print("Frames per second is : {:.0f}".format(fps))

                gsp_video_frames = np.stack(video_frames, axis=0)
                print(gsp_video_frames.shape)
                capture.release()
                # 视频总帧数
                print(f'the number of frames: {frame_count}')
                break  # 当获取完最后一帧就结束
            else:
                video_frames.append(img)
                # cv2.imshow("video", img)  # 用cv2.imshow()查看这一帧，也可以逐帧保存
                # cv2.waitKey(10)  # 必须加等待时间，否则video画面不显示。

            frame_count += 1  # 读取视频帧数＋1
    else:
        print('Video Open Failed!')

    # return video_frames, frame_count
    return gsp_video_frames, frame_count


def Pic2Save(video_frames, dst_folder):
    idx = 0
    # for item in frame:
    for item in video_frames:
        # 设置保存文件名
        save_path = "{}/{:>03d}.jpg".format(dst_folder, idx)
        # 保存图片
        cv2.imwrite(save_path, item)
        idx += 1  # 保存图片数＋1
        print(idx)


# fps: 帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次]
# 如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
# 图片合成视频 https://blog.csdn.net/awujiang/article/details/105402747
# https://blog.csdn.net/caimengxin/article/details/119785199 是将某个文件夹中的所有图片，合成为一个视频。
# https://blog.csdn.net/mao_hui_fei/article/details/107573021
def Pic2Video(video_path, output_path, fps, size):
    filelist = os.listdir(video_path)  # 获取该目录下的所有文件名
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # video = cv2.VideoWriter("output2025.mp4", fourcc, fps, size)
    video = cv2.VideoWriter(output_path, fourcc, fps, size)
    idx = 0
    for f1 in filelist:
        idx += 1
        item = os.path.join(video_path, f1)   # item = path + '/' + f1
        img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR，注意是BGR，通道值默认范围0-255。
        video.write(img)  # 把图片写进视频
        print(idx)

    video.release()  # 释放


def Video_Playing(video_path):
    capture = cv2.VideoCapture(video_path)
    frame_count = 0
    if capture.isOpened():
        while True:
            ret, img = capture.read()  # img 就是一帧图片
            if not ret:
                print('Video Play Out!')
                fps = capture.get(cv2.CAP_PROP_FPS)  # 获取当前版本opencv的FPS
                print("Frames per second is : {}".format(fps))
                capture.release()
                # 视频总帧数
                print(f'the number of frames: {frame_count}')
                break  # 当获取完最后一帧就结束
            else:
                cv2.imshow("video", img)  # 用cv2.imshow()查看这一帧，也可以逐帧保存
                key = cv2.waitKey(10)  # 等待一段时间，并且检测键盘输入
                if key == ord('q'):  # 若是键盘输入'q',则退出，释放视频
                    capture.release()  # 释放视频
                    break
            frame_count += 1  # 读取视频帧数＋1
    else:
        print('Video Open Failed!')


# cv2.VideoWriter_fourcc('X', '2', '6', '4'), 该参数是较新的MPEG-4编码,产生的文件较小,文件扩展名应为.mp4
# cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'), 该参数是较旧的MPEG-1编码,文件名后缀为.avi
# cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 该参数是MPEG-2编码,产生的文件不会特别大,文件名后缀为.avi
# cv2.VideoWriter_fourcc('D', 'I', 'V', '3'), 该参数是MPEG-3编码,产生的文件不会特别大,文件名后缀为.avi
# cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 该参数是MPEG-4编码,产生的文件不会特别大,文件名后缀为.avi
# cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 该参数是较旧的MPEG-4编码,产生的文件不会特别大,文件名后缀为.avi
# cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 该参数也是较旧的MPEG-4编码,产生的文件不会特别大,文件扩展名应为.m4v
# cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'), 该参数是Ogg Vorbis,产生的文件相对较大,文件名后缀为.ogv
# cv2.VideoWriter_fourcc('F', 'L', 'V', '1'), 该参数是Flash视频,产生的文件相对较大,文件名后缀为.flv
# cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 该参数是motion-jpeg编码,产生的文件较大,文件名后缀为.avi
# cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是未压缩的YUV编码,4:2:0色度子采样,这种编码广泛兼容,但会产生特别大的文件,文件扩展名应为.avi ```
