This markdown file records my exploration of using python to develop Android apps, including kivy, beeware and Android Studio. \
It also includes problems I encountered when using beeware and the corresponding solutions, and some tricks of beeware, which seem basic but really confusing to me when I first started using beeware.

# Kivy
It can package your Python project. But I had some problems with it that I didn't solve.\
And it's been a few days since I tried kivy, so I can't remember the exact problems. Sorry about that.

# Beeware
Beeware is the tool that I solved all the problems for my needs and ended up using.
1. First, I followed the [Beeware Tutorial](https://docs.beeware.org/en/latest/), it's the first step to get familar with beeware.
2. The first time you build your project, i.e. the first time you run the code <code>briefcase build android</code>, it might takes some time. It usually takes 10 minutes. Worse, it can get stuck. I got stuck with a gradle file, which I solved by manually downloading the zip form the given link and putting it into the specific path.
- You might get an error like "zip END header not found", which may be caused by an incomplete file, or you can manually download the file  and put it into the specific path.
3. What should you do?
- Your app logic should be completed in the file <code>\src\yourProjectName\resources\app.py</de>.
- If you use the third party, you should add it into the file <code>pyproject.toml</code>. 下面是例子
```
requires=[
  "pillow",
  "httpx==0.27.2"
]
```
- 在toml中可以指定引入库的版本号，特别是当你运行<code>briefcase build android</code>的时候发现 "httpx Error can not find a statisfy version of the httpx"（大概意思是这样，我不记得具体是什么）. 因为你已经运行<code>briefcase dev</code>成功了（我默认是这样，因为通常先执行这句测试成功才去创建apk），所以可以在terminal查看这个库的版本在toml中指定
4. create&update
- 更改pyproject.toml后要重新执行<code>briefcase create android</code>,可以重新生成<code>\build\yourProjectName\android\gradle\app\requirements.txt</code>
- 更改app.py后，通常需要执行<code>briefcase update android</code>
5. 关于调试
- 打开手机的开发者模式（自行上网搜索，不同手机应该不一样吧）
- 打开无线调试（设置->系统与更新->开发人员选项->无线调试）
- 运行<code>briefcase run android</code>，选择emulater，会在手机弹出窗口下载安装
6. 与Android Studio
- 都是用了gradle 和 chaquo吧，但是Android Studio想要用python开发
- beeware真的很省心了，要配置的东西很少
- beeware真的是Python友好，直接用python不需要与Java互调🤤
- 但是beeware设计相对简单吧

- - - - - - - - 
一些对我有参考价值的文章
https://cloud.tencent.cn/developer/article/2369027?from=15425
