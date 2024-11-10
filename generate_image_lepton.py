# import io
# from io import BytesIO
# import requests
# from urllib.request import urlopen

# from leptonai.photon import Photon, PNGResponse


# # 需要继承 Photon 类
# class Canny(Photon):
#     """Canny 边缘检测算子"""

#     # 这里的依赖 Package 会在创建 Photon 时自动安装
#     requirement_dependency = [
#         "numpy",
#         "Pillow",
#         "httpx==0.27.2",
#         "tensorflow"
#     ]

#     # 用这个装饰器表示这个一个对外接口
#     @Photon.handler("run")
#     def run(self, content_url: str, style_url:str) -> PNGResponse:
#         from PIL import Image
#         #
#         image = Image.open(io.BytesIO(urlopen(content_url).read()))
#         #
#         # # 进行边缘检测
#         # edges = cv2.Canny(image, 100, 200)

#         # edges = style_transfer(content_url, style_url)
#         # edges = Image.fromarray(image)

#         img_io = BytesIO()
#         image.save(img_io, format="PNG", quality="keep")
#         img_io.seek(0)
#         return PNGResponse(img_io)
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
        "torch",
    ]

    # 用这个装饰器表示这个一个对外接口
    @Photon.handler("run")
    def run(self, url: str, style_url: str) -> PNGResponse:
        # 读取图像数据
        import numpy as np
        from PIL import Image

        image = np.asarray(Image.open(io.BytesIO(urlopen(url).read())))
        # image = Image.open(io.BytesIO(urlopen(url).read()))

        # # 进行边缘检测
        # edges = cv2.Canny(image, 100, 200)

        # edges = Image.fromarray(edges)
        edges = Image.fromarray(image)

        img_io = BytesIO()
        edges.save(img_io, format="PNG", quality="keep")
        img_io.seek(0)
        return PNGResponse(img_io)
