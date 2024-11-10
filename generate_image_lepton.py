import io
from io import BytesIO
import requests
from urllib.request import urlopen

from leptonai.photon import Photon, PNGResponse


# 需要继承 Photon 类
class Canny(Photon):
    """Canny 边缘检测算子"""

    # 这里的依赖 Package 会在创建 Photon 时自动安装
    requirement_dependency = [
        "numpy",
        "Pillow",
        "httpx==0.27.2",
        "tensorflow"
    ]

    # 用这个装饰器表示这个一个对外接口
    @Photon.handler("run")
    def run(self, content_url: str, style_url:str) -> PNGResponse:
        import os
        import numpy as np

        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.applications import vgg19

        # Generated image size
        RESIZE_HEIGHT = 607

        # NUM_ITER = 3000
        NUM_ITER = 3

        # Weights of the different loss components
        CONTENT_WEIGHT = 8e-4  # 8e-4
        STYLE_WEIGHT = 8e-1  # 8e-1

        # The layer to use for the content loss.
        CONTENT_LAYER_NAME = "block5_conv2"  # "block2_conv2"

        # List of layers to use for the style loss.
        STYLE_LAYER_NAMES = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]

        def get_result_image_size(image_path, result_height):
            '''
            :param image_path: 要调整的图像
            :param result_height: 期待的图像高度
            :return: 对应的高度和宽度
            '''
            image_width, image_height = keras.preprocessing.image.load_img(image_path).size
            result_width = int(image_width * result_height / image_height)
            return result_height, result_width

        def preprocess_image(image_path, target_height, target_width):
            img = keras.preprocessing.image.load_img(image_path, target_size=(target_height, target_width))
            arr = keras.preprocessing.image.img_to_array(img)
            # 增加维度 以符合模型的输入 (a,b,c)->(1,a,b,c)
            arr = np.expand_dims(arr, axis=0)
            # 为符合VGG19的输入，使用vgg19的预处理函数进行预处理
            arr = vgg19.preprocess_input(arr)
            # 结果以tensor张量形式返回
            return tf.convert_to_tensor(arr)

        def get_model():
            # Build a VGG19 model loaded with pre-trained ImageNet weights
            model = vgg19.VGG19(weights='imagenet', include_top=False)  # 加载预训练模型，不要顶层全连接层，以获得中间层输出

            # Get the symbolic outputs of each "key" layer (we gave them unique names).
            outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])  # 获得中间层输出

            # Set up a model that returns the activation values for every layer in VGG19 (as a dict).
            # 构建一个新模型， 输入为VGG19原始输入，输出为VGG19各层输出
            return keras.Model(inputs=model.inputs, outputs=outputs_dict)

        def get_optimizer():
            return keras.optimizers.Adam(  # 构建Adam优化器
                keras.optimizers.schedules.ExponentialDecay(  # 学习率调度器，随时间变化降低学习率，更稳定地收敛
                    initial_learning_rate=8.0, decay_steps=445, decay_rate=0.98  # 初始学习率8， 每445步降低学习率，学习率以0.98衰减
                    # initial_learning_rate = 2.0, decay_steps = 376, decay_rate = 0.98
                )
            )

        def compute_loss(feature_extractor, combination_image, content_features, style_features):
            '''
            :param feature_extractor: model
            :param combination_image: generated_image
            :param content_features: content_features
            :param style_features: style_features
            :return:
            '''
            combination_features = feature_extractor(combination_image)  # 计算当前生成图像的feature， 初始是 white noise image
            loss_content = compute_content_loss(content_features, combination_features)
            # combination~(batch_size, width, height, channels)-> shape[1]*shape[2] is size of feature map, corresponds to Ml in paper.
            loss_style = compute_style_loss(style_features, combination_features,
                                            combination_image.shape[1] * combination_image.shape[2])

            # content_wight/style_weight = 1e-3
            return CONTENT_WEIGHT * loss_content + STYLE_WEIGHT * loss_style

        # A loss function designed to maintain the 'content' of the original_image in the generated_image
        def compute_content_loss(content_features, combination_features):
            '''
            :param content_features: feature of content image
            :param combination_features: feature of combination image
            :return: content loss
            '''
            # CONTENT_LAYER_NAME = "block5_conv2" 用这一层计算content loss 用高层
            original_image = content_features[CONTENT_LAYER_NAME]
            generated_image = combination_features[CONTENT_LAYER_NAME]

            # 均方误差
            return tf.reduce_sum(tf.square(generated_image - original_image)) / 2

        def compute_style_loss(style_features, combination_features, combination_size):
            loss_style = 0

            for layer_name in STYLE_LAYER_NAMES:
                style_feature = style_features[layer_name][0]
                combination_feature = combination_features[layer_name][0]
                # 整体style loss是各层的和
                loss_style += style_loss(style_feature, combination_feature, combination_size) / len(STYLE_LAYER_NAMES)

            return loss_style

        # The "style loss" is designed to maintain the style of the reference image in the generated image.
        # It is based on the gram matrices (which capture style) of feature maps from the style reference image and from the generated image
        def style_loss(style_features, combination_features, combination_size):
            '''
            :param style_features: style image 某一层的特征
            :param combination_features: combination image 相同层的特征
            :param combination_size: Ml size of feature map of combination image
            :return: style loss of one layer
            '''
            S = gram_matrix(style_features)  # 计算Gram矩阵
            C = gram_matrix(combination_features)
            channels = style_features.shape[2]  # 通道数, 这里是三维，因为传进来之前style_features[layer_name][0]
            return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (combination_size ** 2))

        def gram_matrix(x):
            x = tf.transpose(x, (2, 0, 1))
            features = tf.reshape(x, (tf.shape(x)[0], -1))
            # inner product
            gram = tf.matmul(features, tf.transpose(features))
            return gram

        def save_result(generated_image, result_height, result_width, name):
            img = deprocess_image(generated_image, result_height, result_width)
            keras.preprocessing.image.save_img(name, img)
            return img

        # Util function to convert a tensor into a valid image
        def deprocess_image(tensor, result_height, result_width):
            tensor = tensor.numpy()
            tensor = tensor.reshape((result_height, result_width, 3))

            # Remove zero-center by mean pixel
            tensor[:, :, 0] += 103.939
            tensor[:, :, 1] += 116.779
            tensor[:, :, 2] += 123.680

            # 'BGR'->'RGB'
            tensor = tensor[:, :, ::-1]
            return np.clip(tensor, 0, 255).astype("uint8")

        # if __name__ == "__main__":
        def style_transfer(content_image, style_image):
            image = Image.open(io.BytesIO(urlopen(content_image).read()))
            return image

            # Prepare content, stlye images
            path = os.path.abspath(os.getcwd())  # 绝对路径
            content_image_path = keras.utils.get_file(path + '\dataset\paris.jpg',
                                                      content_image)  # 下载图片 保存
            style_image_path = keras.utils.get_file(path + '\dataset\starry_night.jpg',
                                                    style_image)
            # content_image_path = keras.utils.get_file(path + '\dataset\paris.jpg',
            #                                           'https://i.imgur.com/F28w3Ac.jpg')  # 下载图片 保存
            # style_image_path = keras.utils.get_file(path + '\dataset\starry_night.jpg',
            #                                         'https://i.imgur.com/9ooB60I.jpg')
            result_height, result_width = get_result_image_size(content_image_path, RESIZE_HEIGHT)  # 将图像调整为目标高度
            print("result resolution: (%d, %d)" % (result_height, result_width))

            # Preprocessing
            content_tensor = preprocess_image(content_image_path, result_height,
                                              result_width)  # 对content image进行处理，以符合VGG19的输入
            style_tensor = preprocess_image(style_image_path, result_height, result_width)
            generated_image = tf.Variable(tf.random.uniform(style_tensor.shape,
                                                            dtype=tf.dtypes.float32))  # 生成一个与style_tensor相同维度的张量，tf.Various将随机张量封装为TensorFlow变量，便于训练和优化
            # generated_image = tf.Variable(preprocess_image(content_image_path, result_height, result_width))

            # Build model
            model = get_model()  # model 输入是VGG19原始输入，输出是VGG19各中间层输出
            optimizer = get_optimizer()  # 构造的学习率随时间变化的Adam优化器
            print(model.summary())

            # 先计算好了再去拟合white noise image
            content_features = model(content_tensor)
            style_features = model(style_tensor)

            # Optimize result image
            for iter in range(NUM_ITER):
                with tf.GradientTape() as tape:
                    '''
                    在 with 语句内的代码块中，任何对变量的操作（如前向传播、损失计算）都会被记录。
                    当调用 tape.gradient(...) 时，GradientTape 会利用记录的信息自动计算损失函数相对于输入（如生成图像）的梯度。

                    在神经网络训练中，通常会在 GradientTape 中进行前向传播，计算损失，并随后使用 tape.gradient 来获取损失相对于模型参数或输入的梯度，以便更新这些参数。
                    '''
                    loss = compute_loss(model, generated_image, content_features, style_features)

                # 计算loss对generated_image的梯度
                # generated_image 是当前的图像？ 是x?
                grads = tape.gradient(loss, generated_image)

                print("iter: %4d, loss: %8.f" % (iter, loss))
                # 根据梯度调整变量的值 根据grads对generated_image更新
                optimizer.apply_gradients([(grads, generated_image)])

                if (iter + 1) % 100 == 0:
                    name = "result/generated_at_iteration_%d.png" % (iter + 1)
                    # save_result(generated_image, result_height, result_width, name)

            name = "result/result_%d_%f_%f.png" % (NUM_ITER, CONTENT_WEIGHT, STYLE_WEIGHT)
            img = save_result(generated_image, result_height, result_width, name)
            return img

        # # 读取图像数据
        # import cv2
        # import numpy as np
        from PIL import Image
        #
        # image = np.asarray(Image.open(io.BytesIO(urlopen(url).read())))
        #
        # # 进行边缘检测
        # edges = cv2.Canny(image, 100, 200)

        edges = style_transfer(content_url, style_url)
        edges = Image.fromarray(edges)

        img_io = BytesIO()
        edges.save(img_io, format="PNG", quality="keep")
        img_io.seek(0)
        return PNGResponse(img_io)