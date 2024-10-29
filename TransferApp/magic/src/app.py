"""
My first application
"""
import os

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from toga import widgets, ScrollContainer
from toga import ImageView, Button, Box, TextInput, Label
from PIL import Image as PILImage
import httpx
from io import BytesIO
import asyncio

# from magic.generate_image import style_transfer

class HelloWorld(toga.App):
    def startup(self):
        """Construct and show the Toga application.

        Usually, you would add your application to a main content box.
        We then create a main window (with a name matching the app), and
        show the main window.
        """
        self.main_window = toga.MainWindow(title="Image Style Transfer")
        #
        self.scroll_container = toga.ScrollContainer()

        # https://s2.loli.net/2024/10/27/2q3CuDpmNXcdnFy.jpg
        self.content_input = TextInput(placeholder="请输入content image url")
        # https://s2.loli.net/2024/10/27/3ABuE9qyznoI1fX.jpg
        self.style_input = TextInput(placeholder="请输入 style image url")

        self.content_button = Button("点击获取content image", on_press=self.on_button_press)
        self.style_button = Button("点击获取style image", on_press=self.on_style_button_press)

        self.content_reminder_label = Label("尚未点击按钮", style=Pack(padding=10))
        self.style_reminder_label = Label("尚未点击按钮", style=Pack(padding=10))

        self.content_confirm_label = Label("", style=Pack(padding=10))
        self.style_confirm_label = Label("", style=Pack(padding=10))

        self.content_image_view = ImageView()
        self.style_image_view = ImageView()

        self.generate_button = Button("点击生成图片", on_press=self.generate_image)
        self.generate_image_view = ImageView()
        self.generate_reminder_label = Label("", style=Pack(padding=10))

        ## 布局
        self.main_box = Box(style=Pack(direction = COLUMN, padding=10))

        # content image 组件放一一列
        self.content_box = Box(style=Pack(direction=COLUMN, padding = 10))
        self.content_box.add(Label("content image url:"))
        self.content_box.add(self.content_input)
        self.content_box.add(self.content_button)
        self.content_box.add(self.content_reminder_label)
        self.content_box.add(self.content_image_view)
        self.content_box.add(self.content_confirm_label)

        # style image 组件放一列
        self.style_box = Box(style=Pack(direction=COLUMN, padding=10))
        self.style_box.add(Label("style image url:"))
        self.style_box.add(self.style_input)
        self.style_box.add(self.style_button)
        self.style_box.add(self.style_reminder_label)
        self.style_box.add(self.style_image_view)
        self.style_box.add(self.style_confirm_label)

        self.generate_box = Box(style=Pack(direction=COLUMN, padding=10))
        self.generate_box.add(self.generate_button)
        self.generate_box.add(self.generate_image_view)
        self.generate_box.add(self.generate_reminder_label)

        self.main_box.add(self.content_box)
        self.main_box.add(self.style_box)
        self.main_box.add(self.generate_box)

        self.scroll_container.content=self.main_box
        self.main_window.content = self.scroll_container
        self.main_window.show()

    def on_button_press(self, widgets):
        self.content_reminder_label.text="点击成功了"
        # 在事件处理函数中运行异步方法
        asyncio.create_task(self.get_img_from_url())

    async def get_img_from_url(self):
        image_url = self.content_input.value
        # image_url = "https://www.quazero.com/uploads/allimg/140303/1-140303214Q2.jpg"
        self.content_reminder_label.text = "点击成功 请稍后"

        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            # 检查请求是否成功
            if response.status_code == 200:
                # print(response.content)
                image_content = PILImage.open(BytesIO(response.content)).convert('RGB').resize((325, 260))
                self.content_image_view.image = image_content
                print(type(image_content))
                # image_content.show()
                self.content_confirm_label.text = "图片获取成功"
            else:
                self.content_confirm_label.text = f"请求失败，状态码: {response.status_code}"

    def on_style_button_press(self, widgets):
        self.style_reminder_label.text="点击成功了"
        # 在事件处理函数中运行异步方法
        asyncio.create_task(self.get_style_img_from_url())

    async def get_style_img_from_url(self):
        image_url = self.style_input.value
        # image_url = "https://www.quazero.com/uploads/allimg/140303/1-140303214Q2.jpg"
        self.style_reminder_label.text = "点击成功 请稍后"

        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            # 检查请求是否成功
            if response.status_code == 200:
                # print(response.content)
                image_content = PILImage.open(BytesIO(response.content)).convert('RGB').resize((325, 260))
                self.style_image_view.image = image_content
                print(type(image_content))
                # image_content.show()
                self.style_confirm_label.text = "图片获取成功"
            else:
                self.style_confirm_label.text = f"请求失败，状态码: {response.status_code}"

    def generate_image(self, widgets):
        # 不生成了 直接设置为content image
        self.generate_image_view.image = toga.Image("generated_image.jpg")
        # self.generate_image_view.image = toga.Image(style_transfer(self.content_image_view.image, self.style_image_view.image))
        self.generate_reminder_label.text = "生成图片成功！"

def main():
    return HelloWorld()
