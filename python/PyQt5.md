# PyQt5

## 概述

`PyQt5` 是 `Digia`的一套 `Qt5` 应用框架与 `python` 的结合，同时支持 `python2.x和 python3.x`。

## 安装

* 安装PyQt5

```sh
pip install PyQt5 -i https://pypi.douban.com/simple
```

* 安装PyQt5-tools

```sh
pip install PyQt5-tools -i https://pypi.douban.com/simple
```

* 配置环境变量

在系统变量**Path**中添加**F:\a_repo\envs\pyqt\Lib\site-packages\pyqt5_tools**

## 样例

```python
import sys
from PyQt5.QtWidgets import QWidget, QApplication

app = QApplication(sys.argv)
widget = QWidget()
widget.resize(640, 480)
widget.setWindowTitle("Hello, PyQt5!")
widget.show()
sys.exit(app.exec())
```

## 配置

* **Qt designer**

```txt
program:F:\a_repo\envs\pyqt\Lib\site-packages\qt5_applications\Qt\bin\designer.exe

Arguments:

Working directory:$FileDir$
```

* **py UIC**

```txt
program:F:\a_repo\envs\pyqt\Scripts\pyuic5.exe

Arguments:$FileName$ -o $FileNameWithoutExtension$.py

Working directory:$FileDir$
```

* **py rcc**

```
program:F:\a_repo\envs\pyqt\Scripts\pyrcc5.exe

Arguments:$FileName$ -o $FileNameWithoutExtension$_rc.py

Working directory:$FileDir$
```

* 使用**External tool**工具转化成py文件，然后新建程序引入包即可

```python
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import index #导入QtTest文件  

if __name__ == '__main__':
    #获取UIC窗口操作权限
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    #调自定义的界面（即刚转换的.py对象）
    Ui = index.Ui_MainWindow() #这里也引用了一次helloworld.py文件的名字注意
    Ui.setupUi(MainWindow)
    #显示窗口并释放资源
    MainWindow.show()
    sys.exit(app.exec_())
```

